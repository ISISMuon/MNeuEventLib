use crate::data::save::utils::*;

use anyhow::Result;
use ndarray::{Array, Dimension};

use hdf5::{
    types::{
        FixedAscii, FixedUnicode, FloatSize, IntSize, TypeDescriptor, VarLenAscii, VarLenUnicode,
    },
    Group, H5Type, Location,
};

use seq_macro::seq;

/// A method for copying fixed-length ascii attributes.
/// Change the upper bound if there are longer fixed-length ascii attributes.
fn copy_fixed_ascii(
    source: &Location,
    target: &Location,
    attr_name: &str,
    len: usize,
) -> anyhow::Result<()> {
    seq!(N in 1..=128 {
        match len {
            #(N => copy_attribute::<FixedAscii<N>>(source, target, attr_name),)*
            _ => anyhow::bail!(
                "copy_attr: fixed-ascii length {len} for '{attr_name}' exceeds the supported range"
            ),
        }
    })
}

/// A method for copying fixed-length unicode attributes.
/// Change the upper bound if there are longer fixed-length unicode attributes.
fn copy_fixed_unicode(
    source: &Location,
    target: &Location,
    attr_name: &str,
    len: usize,
) -> anyhow::Result<()> {
    seq!(N in 1..=128 {
        match len {
            #(N => copy_attribute::<FixedUnicode<N>>(source, target, attr_name),)*
            _ => anyhow::bail!(
                "copy_attr: fixed-unicode length {len} for '{attr_name}' exceeds the supported range"
            ),
        }
    })
}

/// A method to copy an attribute of statically-known type `T`,
/// while preserving the attribute name.
///
/// Parameters
/// ----------
/// source: &Location
///    The source group
/// target: &Location
///    The destination group
/// attr_name: &str
///    The name of the attribute to copy
pub fn copy_attribute<T: H5Type + Clone>(
    source: &Location,
    target: &Location,
    attr_name: &str,
) -> Result<()> {
    let src_attr = source.attr(attr_name)?;

    if src_attr.is_scalar() {
        let value: T = src_attr.read_scalar()?;
        target
            .new_attr::<T>()
            .create(attr_name)?
            .write_scalar(&value)?;
    } else {
        let data: ndarray::ArrayD<T> = src_attr.read_dyn()?;
        target
            .new_attr::<T>()
            .shape(src_attr.shape())
            .create(attr_name)?
            .write(&data)?;
    }
    Ok(())
}

/// A helper method to copy an attribute of dynamically-known type,
/// while preserving the attribute name.
/// The ascii and unicode values cannot be recorded as variable length,
/// as this causes the programme to crash. This is because HDF5
/// does not register between fixed and variable length
/// string types.
/// Parameters
/// ----------
/// source: &Location
///    The source group
/// target: &Location
///    The destination group
/// attr_name: &str
///    The name of the attribute to copy
pub fn copy_attr(source: &Location, target: &Location, attr_name: &str) -> Result<()> {
    if target.attr(attr_name).is_ok() {
        return Ok(());
    }
    let src_attr = source.attr(attr_name)?;
    let desc = src_attr.dtype()?.to_descriptor()?;

    macro_rules! copy_as {
        ($t:ty) => {
            copy_attribute::<$t>(source, target, attr_name)
        };
    }
    match desc {
        TypeDescriptor::Integer(IntSize::U1) => copy_as!(i8),
        TypeDescriptor::Integer(IntSize::U2) => copy_as!(i16),
        TypeDescriptor::Integer(IntSize::U4) => copy_as!(i32),
        TypeDescriptor::Integer(IntSize::U8) => copy_as!(i64),
        TypeDescriptor::Unsigned(IntSize::U1) => copy_as!(u8),
        TypeDescriptor::Unsigned(IntSize::U2) => copy_as!(u16),
        TypeDescriptor::Unsigned(IntSize::U4) => copy_as!(u32),
        TypeDescriptor::Unsigned(IntSize::U8) => copy_as!(u64),
        TypeDescriptor::Float(FloatSize::U4) => copy_as!(f32),
        TypeDescriptor::Float(FloatSize::U8) => copy_as!(f64),
        TypeDescriptor::Boolean => copy_as!(bool),
        TypeDescriptor::VarLenAscii => copy_as!(VarLenAscii),
        TypeDescriptor::VarLenUnicode => copy_as!(VarLenUnicode),
        TypeDescriptor::FixedAscii(len) => copy_fixed_ascii(source, target, attr_name, len),
        TypeDescriptor::FixedUnicode(len) => copy_fixed_unicode(source, target, attr_name, len),
        other => {
            anyhow::bail!("copy_attr: unsupported type for attribute '{attr_name}': {other:?}")
        }
    }
}

/// Replaces a dataset with a new one, copying all attributes from
//  the old dataset to the new one.
///
/// Parameters
/// ----------
/// group: &Group
///    The group containing the dataset
/// name: &str
///    The name of the dataset to replace
/// new_data: &Array<T, D>
///    The new data to replace the dataset with
pub fn replace_dataset<T: H5Type, D: Dimension>(
    group: &Group,
    name: &str,
    new_data: &Array<T, D>,
) -> Result<()> {
    let old = group.dataset(name)?;
    let attr_names = old.attr_names()?;
    let tmp_name = format!("{name}__tmp");
    let new_ds = group
        .new_dataset::<T>()
        .shape(new_data.shape())
        .create(tmp_name.as_str())?;
    new_ds.write(new_data)?;

    for attr_name in &attr_names {
        copy_attr(&old, &new_ds, attr_name)?;
    }

    drop(old);
    group.unlink(name)?;
    group.relink(tmp_name.as_str(), name)?;

    Ok(())
}

/// Cleans up a string dataset by replacing empty strings with "Missing".
///
/// Parameters
/// ----------
/// group: &Group
///    The group containing the dataset
/// name: &str
///    The name of the dataset to clean
pub fn clean_str_dataset<const LEN: usize>(group: &Group, name: &str) -> Result<()> {
    let is_scalar = group.dataset(name)?.shape().is_empty();
    let value: &hdf5::types::VarLenUnicode = if is_scalar {
        &group.dataset(name)?.read_scalar()?
    } else {
        &group.dataset(name)?.read_1d()?[0]
    };
    if value.as_str() == "" {
        replace_str_dataset::<7>(group, name, "Missing", "")?;
    } else {
        replace_str_dataset::<LEN>(group, name, value.as_str(), value.as_str())?;
    }
    Ok(())
}

/// Replaces a string dataset with a new one,
/// copying all attributes from the old dataset to the new one.
///
/// Parameters
/// ----------
/// group: &Group
///    The group containing the dataset
/// name: &str
///    The name of the dataset to replace
/// new_value: &str
///    The new value to replace the dataset with
/// bad_value: &str
///    The value to replace (ignores if the current value is
///    not equal to `bad_value`)
pub fn replace_str_dataset<const LEN: usize>(
    group: &Group,
    name: &str,
    new_value: &str,
    bad_value: &str,
) -> Result<()> {
    let is_scalar = group.dataset(name)?.shape().is_empty();
    let value: &hdf5::types::VarLenUnicode = if is_scalar {
        &group.dataset(name)?.read_scalar()?
    } else {
        &group.dataset(name)?.read_1d()?[0]
    };
    if value.as_str() != bad_value {
        return Ok(());
    }
    // collect attributes
    let dataset = group.dataset(name)?;
    let attr_names = dataset.attr_names()?;
    group.unlink(name)?;
    add_str_scalar::<LEN>(group, new_value, name)?;
    let new_ds = group.dataset(name)?;
    for attr_name in &attr_names {
        copy_attr(&dataset, &new_ds, attr_name)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hdf5::types::{FixedAscii, VarLenUnicode};
    use hdf5::File;
    use ndarray::{arr0, Array1};
    use std::str::FromStr;
    use tempfile::tempdir;

    fn create_test_file(
        name: &str,
    ) -> (tempfile::TempDir, File, std::sync::MutexGuard<'static, ()>) {
        // HDF5 files can't be read from while they are being written to.
        // This helper method uses the crate's global mutex to prevent other threads from writing
        // to the file at the same time
        let guard = crate::test_utils::lock_hdf5_test();
        let dir = tempdir().unwrap();
        let path = dir.path().join(format!("{name}.nxs"));
        let file = File::create(&path).unwrap();
        (dir, file, guard)
    }

    #[test]
    fn test_copy_attr_scalar() {
        // Tests that a scalar attribute is copied correctly
        let (_dir, file, _guard) = create_test_file("test_copy_attr_scalar");
        let src = file.create_group("src").unwrap();
        let dst = file.create_group("dst").unwrap();

        add_attr(&src, 42i32, "my_int").unwrap();
        copy_attr(&src, &dst, "my_int").unwrap();

        let val: i32 = dst.attr("my_int").unwrap().read_scalar().unwrap();
        assert_eq!(val, 42);
    }

    #[test]
    fn test_copy_attr_str() {
        // Tests that a string attribute is copied correctly
        let (_dir, file, _guard) = create_test_file("test_copy_attr_str");
        let src = file.create_group("src").unwrap();
        let dst = file.create_group("dst").unwrap();

        add_str_attr::<10>(&src, "muon", "particle").unwrap();
        copy_attr(&src, &dst, "particle").unwrap();

        let val: FixedAscii<10> = dst.attr("particle").unwrap().read_scalar().unwrap();
        assert_eq!(val.as_str(), "muon");
    }

    #[test]
    fn test_copy_attr_already_exists_skipped() {
        // Tests that if the attribute already exists, it is not overwritten
        let (_dir, file, _guard) = create_test_file("test_copy_attr_already_exists");
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
        // Tests that if the attribute does not exist, it returns an error
        let (_dir, file, _guard) = create_test_file("test_copy_attr_nonexistent");
        let src = file.create_group("src").unwrap();
        let dst = file.create_group("dst").unwrap();

        let res = copy_attr(&src, &dst, "does_not_exist");
        assert!(res.is_err());
    }

    #[test]
    fn test_replace_dataset() {
        // Tests that a dataset's values can be replaced correctly
        // including copying the original dataset's attributes to the 'new' dataset
        let (_dir, file, _guard) = create_test_file("test_replace_dataset");
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
        // Tests that an empty string dataset is replaced with 'Missing'
        // and that the original dataset's attributes are copied to the 'new' dataset
        let (_dir, file, _guard) = create_test_file("test_clean_str_empty_scalar");
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
        // Tests that a non-empty string dataset is not modified
        let (_dir, file, _guard) = create_test_file("test_clean_str_nonempty_scalar");
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
        // Tests that an empty string dataset in a 1D array is replaced with 'Missing'
        // and that the original dataset's attributes are copied to the 'new' dataset
        let (_dir, file, _guard) = create_test_file("test_clean_str_1d");
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
        // Tests that if the dataset contains the bad value, it is replaced with the new value
        // and that the original dataset's attributes are copied to the 'new' dataset
        let (_dir, file, _guard) = create_test_file("test_replace_str_match");
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
        // Tests that if the dataset does not contain the bad value, it is not modified
        let (_dir, file, _guard) = create_test_file("test_replace_str_nomatch");
        let group = file.create_group("grp").unwrap();

        let text = VarLenUnicode::from_str("correct_val").unwrap();
        let _ = add_array(&group, &arr0(text), "tag").unwrap();

        replace_str_dataset::<16>(&group, "tag", "new_val", "different_bad_val").unwrap();

        let ds = group.dataset("tag").unwrap();
        let val: VarLenUnicode = ds.read_scalar().unwrap();
        assert_eq!(val.as_str(), "correct_val");
    }
}
