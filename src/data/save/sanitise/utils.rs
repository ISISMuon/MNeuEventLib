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


pub fn copy_attr(src: &Location, dst: &Location, name: &str) -> Result<()> {
    unsafe { _copy_attr(src, dst, name) };
    Ok(())
}

/// Copies a single attribute from `src` to `dst`, regardless of its HDF5 datatype
/// (ints, floats, fixed/var-length strings, enums, compounds, arrays...).
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
    let attr_names = dataset.attr_names()?;
    group.unlink(name)?;
    add_str_scalar::<LEN>(group, new_value, name)?;
    let new_ds = group.dataset(name).unwrap();
    for attr_name in &attr_names {
        unsafe{ copy_attr(&dataset, &new_ds, attr_name) };
    }
    Ok(())
}
