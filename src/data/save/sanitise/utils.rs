use std::str::FromStr;

use crate::data::save::utils::*;

use anyhow::Result;
use hdf5::types::{FixedAscii, H5Type, VarLenUnicode};
use hdf5::{Dataset, File, Group, Location};
use ndarray::{arr0, Array, Array1, Dimension};


//use hdf5_metno as hdf5;
use hdf5_metno_sys::{h5a, h5p, h5t, h5s, h5d};
use hdf5::{Result as OtherResult};
use std::ffi::CString;
use std::os::raw::c_void;

/// A wrapper around _copy_attr which checks if the 
/// attribute already exists and skips if it does.
///
/// ## Arguments
/// * `src` - The source group
/// * `dst` - The destination group
/// * `name` - The name of the attribute to copy
/// 
/// ## Returns
/// * `Ok(())` - If the attribute is copied successfully
/// * `Err(anyhow::Error)` - If the attribute cannot be copied
pub fn copy_attr(src: &Location, dst: &Location, name: &str) -> Result<()> {
    unsafe { _copy_attr(src, dst, name) };
    Ok(())
}

/// Copies a single attribute from `src` to `dst`, regardless of its HDF5 datatype
/// (ints, floats, fixed/var-length strings, enums, compounds, arrays...).
///
/// ## Arguments
/// * `src` - The source group
/// * `dst` - The destination group
/// * `name` - The name of the attribute to copy
/// 
/// ## Returns
/// * `Ok(())` - If the attribute is copied successfully
/// * `Err(anyhow::Error)` - If the attribute cannot be copied
unsafe fn _copy_attr(src: &Location, dst: &Location, name: &str) -> Result<()> {
    if dst.attr_names()?.contains(&name.to_string()) {
        println!("Attribute '{name}' already exists in destination dataset, skipping");
        return Ok(());
    }
    let cname = CString::new(name).unwrap();

    let attr_id = h5a::H5Aopen(src.id(), cname.as_ptr(), h5p::H5P_DEFAULT);
    if attr_id < 0 {
        return Err(hdf5::Error::from(format!("H5Aopen failed for '{name}'")).into());
    }

    let type_id = h5a::H5Aget_type(attr_id);
    let space_id = h5a::H5Aget_space(attr_id);
    let storage_size = h5a::H5Aget_storage_size(attr_id) as usize;

    let mut buf: Vec<u8> = vec![0u8; storage_size.max(1)];
    h5a::H5Aread(attr_id, type_id, buf.as_mut_ptr() as *mut c_void);

    let new_attr_id = h5a::H5Acreate2(
        dst.id(), cname.as_ptr(), type_id, space_id,
        h5p::H5P_DEFAULT, h5p::H5P_DEFAULT,
    );
    h5a::H5Awrite(new_attr_id, type_id, buf.as_ptr() as *const c_void);

    // Modern replacement for H5Dvlen_reclaim — same purpose, lives in h5t.
    h5t::H5Treclaim(type_id, space_id, h5p::H5P_DEFAULT, buf.as_mut_ptr() as *mut c_void);

    h5a::H5Aclose(new_attr_id);
    h5a::H5Aclose(attr_id);
    h5t::H5Tclose(type_id);
    h5s::H5Sclose(space_id);

    Ok(())
}

/// Replaces a dataset with a new one, copying all attributes from
//  the old dataset to the new one.
///
/// ## Arguments
/// * `group` - The group containing the dataset
/// * `name` - The name of the dataset to replace
/// * `new_data` - The new data to replace the dataset with
///
/// ## Returns
/// * `Ok(())` - If the dataset is replaced successfully
/// * `Err(hdf5::Error)` - If the dataset cannot be replaced
pub fn replace_dataset<T:H5Type, D: Dimension>(
    group: &Group,
    name: &str,
    new_data: &Array<T, D>,
) -> OtherResult<()> {
    let old = group.dataset(name)?;
    let attr_names = old.attr_names()?;
    let tmp_name = format!("{name}__tmp");
    let new_ds = group.new_dataset::<T>().shape(new_data.shape()).create(tmp_name.as_str())?;
    new_ds.write(new_data)?;

    for attr_name in &attr_names {
        unsafe{ copy_attr(&old, &new_ds, attr_name) };
    }

    drop(old);
    group.unlink(name)?;
    group.relink(tmp_name.as_str(), name)?;

    Ok(())
}

/// Cleans up a string dataset by replacing empty strings with "Missing".
///
/// ## Arguments
/// * `group` - The group containing the dataset
/// * `name` - The name of the dataset to clean
///
/// ## Returns
/// * `Ok(())` - If the dataset is cleaned successfully
/// * `Err(anyhow::Error)` - If the dataset cannot be cleaned
pub fn clean_str_dataset<const LEN: usize>(
    group: &Group,
    name: &str,
) -> Result<()> {
    let is_scalar = group.dataset(name).unwrap().shape().is_empty();
    let value: &hdf5::types::VarLenUnicode = if is_scalar{
        &group.dataset(name).unwrap().read_scalar().unwrap()
    }else{
        &group.dataset(name).unwrap().read_1d().unwrap()[0]
    };
    if value.as_str() =="" {
        replace_str_dataset::<7>(group, name, "Missing", "");
    }else{
        replace_str_dataset::<LEN>(group, name, value.as_str(), value.as_str());
    }
    Ok(())
}

/// Replaces a string dataset with a new one,
/// copying all attributes from the old dataset to the new one.
///
/// ## Arguments
/// * `group` - The group containing the dataset
/// * `name` - The name of the dataset to replace
/// * `new_value` - The new value to replace the dataset with
/// * `bad_value` - The value to replace (ignores if the current value is 
///                 not equal to `bad_value`)
///
/// ## Returns
/// * `Ok(())` - If the dataset is replaced successfully
/// * `Err(anyhow::Error)` - If the dataset cannot be replaced
pub fn replace_str_dataset<const LEN: usize>(
    group: &Group,
    name: &str,
    new_value: &str,
    bad_value: &str,
) -> Result<()> {
    let is_scalar = group.dataset(name).unwrap().shape().is_empty();
    let value: &hdf5::types::VarLenUnicode = if is_scalar{
        &group.dataset(name).unwrap().read_scalar().unwrap()
    }else{
        &group.dataset(name).unwrap().read_1d().unwrap()[0]
    };
    if value.as_str() !=bad_value {
        return Ok(())
    }
    // collect attributes
    let dataset = group.dataset(name).unwrap();
    let attr_names = dataset.attr_names()?;
    group.unlink(name)?;
    add_str_scalar::<LEN>(group, new_value, name)?;
    let new_ds = group.dataset(name)?;
    for attr_name in &attr_names {
        unsafe { copy_attr(&dataset, &new_ds, attr_name)? };
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hdf5::File;
    use ndarray::{arr0, Array1};
    use tempfile::tempdir;

    fn create_test_file(name: &str) -> (tempfile::TempDir, File) {
        let dir = tempdir().unwrap();
        let path = dir.path().join(format!("{name}.nxs"));
        let file = File::create(&path).unwrap();
        (dir, file)
    }

    #[test]
    fn test_copy_attr_scalar() {
        let (_dir, file) = create_test_file("test_copy_attr_scalar");
        let src = file.create_group("src").unwrap();
        let dst = file.create_group("dst").unwrap();

        add_attr(&src, 42i32, "my_int").unwrap();
        copy_attr(&src, &dst, "my_int").unwrap();

        let val: i32 = dst.attr("my_int").unwrap().read_scalar().unwrap();
        assert_eq!(val, 42);
    }

    #[test]
    fn test_copy_attr_str() {
        let (_dir, file) = create_test_file("test_copy_attr_str");
        let src = file.create_group("src").unwrap();
        let dst = file.create_group("dst").unwrap();

        add_str_attr::<10>(&src, "muon", "particle").unwrap();
        copy_attr(&src, &dst, "particle").unwrap();

        let val: FixedAscii<10> = dst.attr("particle").unwrap().read_scalar().unwrap();
        assert_eq!(val.as_str(), "muon");
    }

    #[test]
    fn test_copy_attr_already_exists_skipped() {
        let (_dir, file) = create_test_file("test_copy_attr_already_exists");
        let src = file.create_group("src").unwrap();
        let dst = file.create_group("dst").unwrap();

        add_attr(&src, 100i32, "shared").unwrap();
        add_attr(&dst, 200i32, "shared").unwrap();

        copy_attr(&src, &dst, "shared").unwrap();

        // Should keep original dst value (200), not overwritten by src (100)
        let val: i32 = dst.attr("shared").unwrap().read_scalar().unwrap();
        assert_eq!(val, 200);
    }

    #[test]
    fn test_copy_attr_nonexistent_fails() {
        let (_dir, file) = create_test_file("test_copy_attr_nonexistent");
        let src = file.create_group("src").unwrap();
        let dst = file.create_group("dst").unwrap();

        let res = copy_attr(&src, &dst, "does_not_exist");
        assert!(res.is_err());
    }

    #[test]
    fn test_replace_dataset() {
        let (_dir, file) = create_test_file("test_replace_dataset");
        let group = file.create_group("grp").unwrap();

        let original_data = Array1::from_vec(vec![1.0f32, 2.0, 3.0]);
        let ds = add_array(&group, &original_data, "values").unwrap();
        add_str_attr::<5>(&ds, "K", "units").unwrap();

        let new_data = Array1::from_vec(vec![10.0f32, 20.0]);
        replace_dataset(&group, "values", &new_data).unwrap();

        let replaced_ds = group.dataset("values").unwrap();
        let read_data: Array1<f32> = replaced_ds.read_1d().unwrap();
        assert_eq!(read_data, new_data);

        // Attribute should be preserved
        let attr_val: FixedAscii<5> = replaced_ds.attr("units").unwrap().read_scalar().unwrap();
        assert_eq!(attr_val.as_str(), "K");
    }

    #[test]
    fn test_clean_str_dataset_empty_scalar() {
        let (_dir, file) = create_test_file("test_clean_str_empty_scalar");
        let group = file.create_group("grp").unwrap();

        let text = VarLenUnicode::from_str("").unwrap();
        let ds = add_array(&group, &arr0(text), "type").unwrap();
        add_str_attr::<5>(&ds, "note", "desc").unwrap();

        clean_str_dataset::<256>(&group, "type").unwrap();

        let cleaned_ds = group.dataset("type").unwrap();
        let val: FixedAscii<7> = cleaned_ds.read_1d().unwrap()[0];
        assert_eq!(val.as_str(), "Missing");

        let attr_val: FixedAscii<5> = cleaned_ds.attr("desc").unwrap().read_scalar().unwrap();
        assert_eq!(attr_val.as_str(), "note");
    }

    #[test]
    fn test_clean_str_dataset_nonempty_scalar() {
        let (_dir, file) = create_test_file("test_clean_str_nonempty_scalar");
        let group = file.create_group("grp").unwrap();

        let text = VarLenUnicode::from_str("Silicon").unwrap();
        let _ = add_array(&group, &arr0(text), "sample").unwrap();

        clean_str_dataset::<16>(&group, "sample").unwrap();

        let cleaned_ds = group.dataset("sample").unwrap();
        let val: FixedAscii<16> = cleaned_ds.read_1d().unwrap()[0];
        assert_eq!(val.as_str(), "Silicon");
    }

    #[test]
    fn test_clean_str_dataset_1d_array() {
        let (_dir, file) = create_test_file("test_clean_str_1d");
        let group = file.create_group("grp").unwrap();

        let text = VarLenUnicode::from_str("").unwrap();
        let arr = Array1::from_elem(1, text);
        let _ = add_array(&group, &arr, "desc").unwrap();

        clean_str_dataset::<64>(&group, "desc").unwrap();

        let cleaned_ds = group.dataset("desc").unwrap();
        let val: FixedAscii<7> = cleaned_ds.read_1d().unwrap()[0];
        assert_eq!(val.as_str(), "Missing");
    }

    #[test]
    fn test_replace_str_dataset_matching_bad_value() {
        let (_dir, file) = create_test_file("test_replace_str_match");
        let group = file.create_group("grp").unwrap();

        let text = VarLenUnicode::from_str("old_val").unwrap();
        let ds = add_array(&group, &arr0(text), "tag").unwrap();
        add_attr(&ds, 99i32, "code").unwrap();

        replace_str_dataset::<16>(&group, "tag", "new_val", "old_val").unwrap();

        let new_ds = group.dataset("tag").unwrap();
        let val: FixedAscii<16> = new_ds.read_1d().unwrap()[0];
        assert_eq!(val.as_str(), "new_val");

        let code_val: i32 = new_ds.attr("code").unwrap().read_scalar().unwrap();
        assert_eq!(code_val, 99);
    }

    #[test]
    fn test_replace_str_dataset_nonmatching_bad_value() {
        let (_dir, file) = create_test_file("test_replace_str_nomatch");
        let group = file.create_group("grp").unwrap();

        let text = VarLenUnicode::from_str("correct_val").unwrap();
        let _ = add_array(&group, &arr0(text), "tag").unwrap();

        replace_str_dataset::<16>(&group, "tag", "new_val", "different_bad_val").unwrap();

        let ds = group.dataset("tag").unwrap();
        let val: VarLenUnicode = ds.read_scalar().unwrap();
        assert_eq!(val.as_str(), "correct_val");
    }
}

