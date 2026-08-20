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

/// Copies a single attribute from `src` to `dst`, regardless of its HDF5 datatype
/// (ints, floats, fixed/var-length strings, enums, compounds, arrays...).
unsafe fn copy_attr(src: &Dataset, dst: &Dataset, name: &str) -> Result<()> {
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

pub fn ensure_fixed_length_string(group: &Group, name: &str) -> Result<()> {
    let ds = group.dataset(name)?;

    let is_scalar = ds.shape().is_empty();

    let is_variable = {
        let dtype = ds.dtype()?;
        let type_id = dtype.id();

        let class = unsafe { h5t::H5Tget_class(type_id) };
        anyhow::ensure!(class == h5t::H5T_class_t::H5T_STRING, "'{name}' is not a string dataset");

        let result = unsafe { h5t::H5Tis_variable_str(type_id) };
        anyhow::ensure!(result >= 0, "H5Tis_variable_str failed for '{name}'");
        result > 0
    };

    if !is_variable {
        eprintln!("'{name}' is already fixed-length — no-op");
        return Ok(());
    }
    eprintln!("'{name}' is variable-length, converting...");

    let attr_names = ds.attr_names()?;
    let tmp_name = format!("{name}__tmp_fixed");

    // If a previous failed attempt left this behind, clean it up first
    // rather than letting H5Dcreate2 fail on a name collision.
    if group.link_exists(&tmp_name) {
        eprintln!("found leftover '{tmp_name}' from a previous run, removing it");
        group.unlink(&tmp_name)?;
    }

    let (elem_count, max_len, packed): (usize, usize, Vec<u8>) = if is_scalar {
        let value: VarLenUnicode = ds.read_scalar()?;
        let len = value.as_str().len().max(1);
        let mut buf = vec![0u8; len];
        buf[..value.as_str().len()].copy_from_slice(value.as_str().as_bytes());
        (1, len, buf)
    } else {
        let values: ndarray::Array1<VarLenUnicode> = ds.read_1d()?;
        let max_len = values.iter().map(|s| s.as_str().len()).max().unwrap_or(0).max(1);
        let mut buf = vec![0u8; values.len() * max_len];
        for (i, v) in values.iter().enumerate() {
            let bytes = v.as_str().as_bytes();
            buf[i * max_len..i * max_len + bytes.len()].copy_from_slice(bytes);
        }
        (values.len(), max_len, buf)
    };
    eprintln!("packed {elem_count} element(s), max_len={max_len}, buf.len()={}", packed.len());

    unsafe {
        let dtype = ds.dtype()?;
        let fixed_type_id = h5t::H5Tcopy(dtype.id());
        anyhow::ensure!(fixed_type_id >= 0, "H5Tcopy failed");
        drop(dtype);

        anyhow::ensure!(h5t::H5Tset_size(fixed_type_id, max_len) >= 0, "H5Tset_size failed");
        anyhow::ensure!(
            h5t::H5Tset_strpad(fixed_type_id, h5t::H5T_str_t::H5T_STR_NULLPAD) >= 0,
            "H5Tset_strpad failed"
        );

        let space_id = if is_scalar {
            h5s::H5Screate(h5s::H5S_class_t::H5S_SCALAR)
        } else {
            let dims = [elem_count as u64];
            h5s::H5Screate_simple(1, dims.as_ptr(), std::ptr::null())
        };
        anyhow::ensure!(space_id >= 0, "dataspace creation failed");

        let cname = CString::new(tmp_name.as_str())?;
        let new_ds_id = h5d::H5Dcreate2(
            group.id(), cname.as_ptr(), fixed_type_id, space_id,
            h5p::H5P_DEFAULT, h5p::H5P_DEFAULT, h5p::H5P_DEFAULT,
        );
        anyhow::ensure!(new_ds_id >= 0, "H5Dcreate2 failed for '{tmp_name}'");

        let write_status = h5d::H5Dwrite(
            new_ds_id, fixed_type_id, h5s::H5S_ALL, h5s::H5S_ALL,
            h5p::H5P_DEFAULT, packed.as_ptr() as *const c_void,
        );
        anyhow::ensure!(write_status >= 0, "H5Dwrite failed for '{tmp_name}'");
        eprintln!("wrote '{tmp_name}' successfully");

        h5d::H5Dclose(new_ds_id);
        h5s::H5Sclose(space_id);
        h5t::H5Tclose(fixed_type_id);
    }

    let new_ds = group.dataset(tmp_name.as_str())?;
    for attr_name in &attr_names {
        unsafe { copy_attr(&ds, &new_ds, attr_name)? };
    }
    drop(new_ds);
    drop(ds);

    group.unlink(name)?;
    group.relink(tmp_name.as_str(), name)?;
    eprintln!("relinked '{tmp_name}' -> '{name}'");

    Ok(())
}


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
    group.unlink(name)?;
    add_str_scalar::<LEN>(group, new_value, name)?;
    //for attr in attrs {
    //    add_attr(group.dataset(name).unwrap(), attr.1, &attr.0)?;
    //}
    Ok(())
}
