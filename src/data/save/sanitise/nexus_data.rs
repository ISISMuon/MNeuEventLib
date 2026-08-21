use std::collections::HashMap;
use std::str::FromStr;

use anyhow::{anyhow, Result};
use hdf5::types::{FixedAscii, VarLenUnicode};
use hdf5::{Dataset, File, Group, Location};
use ndarray::Array1;
use crate::data::save::sanitise::utils::*;
use crate::data::save::utils::*;

//struct AttrVal {
//    name: String,
//    val_i32: Option<i32>,
//    val_u32: Option<u32>,
//    val_f64: Option<f64>,
//    val_str: Option<VarLenUnicode>,
//}

//fn get_dataset_attrs(ds: &Dataset) -> Result<Vec<AttrVal>> {
//    let mut attrs = Vec::new();
//    for name in ds.attr_names()? {
//        let attr = ds.attr(&name)?;
//        let dtype = attr.dtype()?;
//        let desc = dtype.to_descriptor()?;
//
//        let mut attr_val = AttrVal {
//            name,
//            val_i32: None,
//            val_u32: None,
//            val_f64: None,
//            val_str: None,
//        };
//
//        match desc {
//            hdf5::types::TypeDescriptor::Integer(_) => {
//                attr_val.val_i32 = Some(attr.read_scalar::<i32>()?);
//            }
//            hdf5::types::TypeDescriptor::Unsigned(_) => {
//                attr_val.val_u32 = Some(attr.read_scalar::<u32>()?);
//            }
//            hdf5::types::TypeDescriptor::Float(_) => {
//                attr_val.val_f64 = Some(attr.read_scalar::<f64>()?);
//            }
//            hdf5::types::TypeDescriptor::VarLenUnicode | hdf5::types::TypeDescriptor::FixedAscii(_) => {
//                attr_val.val_str = Some(attr.read_scalar::<VarLenUnicode>()?);
//            }
//            _ => {
//                if let Ok(s) = attr.read_scalar::<VarLenUnicode>() {
//                    attr_val.val_str = Some(s);
//                }
//            }
//        }
//        attrs.push(attr_val);
//    }
//    Ok(attrs)
//}
//
//fn write_dataset_attrs(ds: &Dataset, attrs: Vec<AttrVal>) -> Result<()> {
//    for attr in attrs {
//        if let Some(val) = attr.val_i32 {
//            let mut writer = ds.new_attr::<i32>().shape([]).create(&attr.name)?;
//            writer.write_scalar(&val)?;
//        } else if let Some(val) = attr.val_u32 {
//            let mut writer = ds.new_attr::<u32>().shape([]).create(&attr.name)?;
//            writer.write_scalar(&val)?;
//        } else if let Some(val) = attr.val_f64 {
//            let mut writer = ds.new_attr::<f64>().shape([]).create(&attr.name)?;
//            writer.write_scalar(&val)?;
//        } else if let Some(val) = attr.val_str {
//            let mut writer = ds.new_attr::<VarLenUnicode>().shape([]).create(&attr.name)?;
//            writer.write_scalar(&val)?;
//        }
//    }
//    Ok(())
//}
//
//
//
//fn copy_dataset(from: &Dataset, to: &Group, name: &str) -> Result<()> {
//    let dtype = from.dtype()?;
//    let desc = dtype.to_descriptor()?;
//    let shape = from.shape();
//
//    match desc {
//        hdf5::types::TypeDescriptor::Integer(_) => {
//            if shape.is_empty() {
//                let val = from.read_scalar::<i32>()?;
//                to.new_dataset::<i32>().shape([]).create(name)?.write_scalar(&val)?;
//            } else {
//                let val = from.read_raw::<i32>()?;
//                to.new_dataset::<i32>().shape(shape.clone()).create(name)?.write(&ndarray::Array::from_shape_vec(shape, val)?)?;
//            }
//        }
//        hdf5::types::TypeDescriptor::Unsigned(_) => {
//            if shape.is_empty() {
//                let val = from.read_scalar::<u32>()?;
//                to.new_dataset::<u32>().shape([]).create(name)?.write_scalar(&val)?;
//            } else {
//                let val = from.read_raw::<u32>()?;
//                to.new_dataset::<u32>().shape(shape.clone()).create(name)?.write(&ndarray::Array::from_shape_vec(shape, val)?)?;
//            }
//        }
//        hdf5::types::TypeDescriptor::Float(_) => {
//            if shape.is_empty() {
//                let val = from.read_scalar::<f64>()?;
//                to.new_dataset::<f64>().shape([]).create(name)?.write_scalar(&val)?;
//            } else {
//                let val = from.read_raw::<f64>()?;
//                to.new_dataset::<f64>().shape(shape.clone()).create(name)?.write(&ndarray::Array::from_shape_vec(shape, val)?)?;
//            }
//        }
//        hdf5::types::TypeDescriptor::VarLenUnicode | hdf5::types::TypeDescriptor::FixedAscii(_) => {
//            if shape.is_empty() {
//                let val = from.read_scalar::<VarLenUnicode>()?;
//                to.new_dataset::<VarLenUnicode>().shape([]).create(name)?.write_scalar(&val)?;
//            } else {
//                let val = from.read_raw::<VarLenUnicode>()?;
//                to.new_dataset::<VarLenUnicode>().shape(shape.clone()).create(name)?.write(&ndarray::Array::from_shape_vec(shape, val)?)?;
//            }
//        }
//        _ => {
//            return Err(anyhow!("Unsupported type descriptor for dataset {}: {:?}", name, desc));
//        }
//    }
//
//    let new_ds = to.dataset(name)?;
//    for attr_name in from.attr_names()? {
//        let attr = from.attr(&attr_name)?;
//        copy_attr(&attr, &new_ds, &attr_name)?;
//    }
//
//    Ok(())
//}
//
//fn create_default_dataset(
//    obj: &Group,
//    dest: &Group,
//    name: &str,
//    shapes: &HashMap<String, usize>,
//) -> Result<()> {
//    let shape_ds = obj.dataset("shape")?;
//    let shape_key = if let Ok(s) = shape_ds.read_scalar::<VarLenUnicode>() {
//        s.to_string()
//    } else if let Ok(s) = shape_ds.read_scalar::<FixedAscii<256>>() {
//        s.as_str().to_string()
//    } else {
//        return Err(anyhow!("Could not read shape dataset as string"));
//    };
//
//    let size = *shapes
//        .get(&shape_key)
//        .ok_or_else(|| anyhow!("Shape key {} not found in shapes map", shape_key))?;
//
//    let default_ds = obj.dataset("default")?;
//    let dtype = default_ds.dtype()?;
//    let desc = dtype.to_descriptor()?;
//
//    match desc {
//        hdf5::types::TypeDescriptor::Integer(_) => {
//            let default_val = default_ds.read_scalar::<i32>()?;
//            let data = Array1::from_elem(size, default_val);
//            let ds = dest.new_dataset::<i32>().shape([size]).create(name)?;
//            ds.write(&data)?;
//        }
//        hdf5::types::TypeDescriptor::Unsigned(_) => {
//            let default_val = default_ds.read_scalar::<u32>()?;
//            let data = Array1::from_elem(size, default_val);
//            let ds = dest.new_dataset::<u32>().shape([size]).create(name)?;
//            ds.write(&data)?;
//        }
//        hdf5::types::TypeDescriptor::Float(_) => {
//            let default_val = default_ds.read_scalar::<f64>()?;
//            let data = Array1::from_elem(size, default_val);
//            let ds = dest.new_dataset::<f64>().shape([size]).create(name)?;
//            ds.write(&data)?;
//        }
//        hdf5::types::TypeDescriptor::VarLenUnicode | hdf5::types::TypeDescriptor::FixedAscii(_) => {
//            let default_val = default_ds.read_scalar::<VarLenUnicode>()?;
//            let data = Array1::from_elem(size, default_val);
//            let ds = dest.new_dataset::<VarLenUnicode>().shape([size]).create(name)?;
//            ds.write(&data)?;
//        }
//        _ => {
//            return Err(anyhow!("Unsupported default dataset type {:?}", desc));
//        }
//    }
//
//    let new_ds = dest.dataset(name)?;
//    for attr_name in obj.attr_names()? {
//        let attr = obj.attr(&attr_name)?;
//        copy_attr(&attr, &new_ds, &attr_name)?;
//    }
//
//    Ok(())
//}
//

// need to make a method to replace the dataset


fn create_default_dataset(default: &Group, dest: &Group, name: &str, shapes: &HashMap<String, usize>) -> Result<()> {
    let name = default.name().split("dataset_").last().unwrap().to_string();
    let key: String = default.dataset("shape").unwrap().read_scalar::<hdf5::types::VarLenUnicode>().unwrap().as_str().to_string();
    let len: usize = *shapes.get(&key).unwrap();
    let dtype: &hdf5::types::VarLenUnicode = &default.dataset("dtype").unwrap().read_scalar().unwrap();
    if dtype.as_str() == "int32"{
        let default_value = default.dataset("default").unwrap().read_scalar::<i32>()?;
        add_array(dest, &Array1::from_elem(len, default_value), &name);
    }else if dtype.as_str() == "float32"{
        let default_value = default.dataset("default").unwrap().read_scalar::<f32>()?;
        add_array(dest, &Array1::from_elem(len, default_value), &name);
    }else if dtype.as_str() == "float64"{
        let default_value = default.dataset("default").unwrap().read_scalar::<f64>()?;
        add_array(dest, &Array1::from_elem(len, default_value), &name);
    }else{
        return Err(anyhow!("Unsupported default dataset type {:?}", dtype));
    };
    for att_name in default.attr_names()?{
        let attr = default.attr(&att_name).unwrap();
        let data = dest.dataset(name.as_str()).unwrap();
        println!("copy attribute {}", att_name);
        unsafe {copy_attr(&attr, &data, &att_name)};
    };
    return Ok(());
}

fn set_defaults(
    source_parent: &Group,
    dest: &Group,
    name: &str,
    shapes: &HashMap<String, usize>,
) -> Result<()> {
    if name.contains("dataset_") && dest.dataset(name).is_err() {
        /* we know this is a group that defines a period
        dependent dataset */
        println!("create default {}", name);
        create_default_dataset(&source_parent.group(name).unwrap(), dest, name, shapes)?;
        return Ok(());
    }else if source_parent.dataset(name).is_ok() && dest.dataset(name).is_ok() {
        // if dataset exists in both files
       println!("{} dataset already exists", name);
       for att_name in source_parent.attr_names()?{
           let attr = source_parent.attr(&att_name).unwrap();
           let data = dest.dataset(name).unwrap();
           println!("copy attribute {}", att_name);
           unsafe {copy_attr(&attr, &data, &att_name)};
       };
 
        return Ok(());
    }else if source_parent.dataset(name).is_ok() && dest.dataset(name).is_err() {
        // if dataset exists in source but not in destination
        println!("copy dataset {}", name);
        source_parent.dataset(name)?.copy_to(dest, name)?;
        return Ok(());
    }else if source_parent.group(name).is_ok() && dest.group(name).is_ok() {
        // if group exists in both files
        println!("group {} exists in both files, going deeper", name);
        for member in source_parent.group(name).unwrap().member_names()? {
            set_defaults(&source_parent.group(name).unwrap(),
              &dest.group(name).unwrap(),
              member.as_str(),
               shapes)?;
        }
        return Ok(());
    }else if source_parent.group(name).is_ok() && dest.group(name).is_err() {
        // copy group
        println!("make a copy of group {}", name);
        /* this works for muons as non of the datasets from the missing group have a
        length that depends on the number of periods. */
        source_parent.group(name)?.copy_to(dest, name)?;
        return Ok(());
    }else{
        return Err(anyhow!("{} does not exist", name));
    }
     // take just the last path segment, in case name() returns a full path like "/group/data_foo"
    //let basename = name.rsplit('/').next().unwrap_or(&name.as_str());
    //println!("base {}", basename);
    // strip_prefix only removes it if it's actually there, and only once
    //let new_name = basename.strip_prefix("data_").unwrap_or(basename);   
    //println!("set_defaults: {}, {}, {}", basename, is_dataset, new_name);

//    let dest_keys = dest.member_names()?;
//
//    if let Ok(source_group) = source_parent.group(name) {
//        if !dest_keys.contains(&target_name) {
//            if !is_template {
//                println!("Copy group: {}", target_name);
//                let tmp = dest.create_group(&target_name)?;
//                for attr_name in source_group.attr_names()? {
//                    let attr = source_group.attr(&attr_name)?;
//                    copy_attr(&attr, &tmp, &attr_name)?;
//                }
//
//                for child_name in source_group.member_names()? {
//                    set_defaults(&source_group, &tmp, &child_name, shapes)?;
//                }
//            } else {
//                println!("Create default: {}", target_name);
//                create_default_dataset(&source_group, dest, &target_name, shapes)?;
//            }
//        } else {
//            if let Ok(dest_child_group) = dest.group(&target_name) {
//                for attr_name in source_group.attr_names()? {
//                    let attr = source_group.attr(&attr_name)?;
//                    copy_attr(&attr, &dest_child_group, &attr_name)?;
//                }
//
//                for child_name in source_group.member_names()? {
//                    set_defaults(&source_group, &dest_child_group, &child_name, shapes)?;
//                }
//            } else if let Ok(dest_child_ds) = dest.dataset(&target_name) {
//                for attr_name in source_group.attr_names()? {
//                    let attr = source_group.attr(&attr_name)?;
//                    copy_attr(&attr, &dest_child_ds, &attr_name)?;
//                }
//            }
//        }
//    } else if let Ok(source_dataset) = source_parent.dataset(name) {
//        if dest_keys.contains(&target_name) {
//            if let Ok(dest_child_ds) = dest.dataset(&target_name) {
//                for attr_name in source_dataset.attr_names()? {
//                    let attr = source_dataset.attr(&attr_name)?;
//                    copy_attr(&attr, &dest_child_ds, &attr_name)?;
//                }
//            }
//        } else {
//            println!("Copy dataset: {}", target_name);
//            copy_dataset(&source_dataset, dest, &target_name)?;
//        }
//    }
//
    Ok(())
}

fn clean_up(new_file: &File) -> Result<()> {
    let hist_data = new_file.group("raw_data_1")?;

    let _ = replace_str_dataset::<4>(&hist_data, "name", "HIFI", "name")?;
    let _ = replace_str_dataset::<4>(&hist_data, "title", "Data", "");


    let sample = hist_data.group("sample")?;
    for name in sample.member_names()? {
        if let Ok(_group) = sample.group(&name) {
            continue;
        } else if let Ok(dataset) = sample.dataset(&name) {
            if name == "thickness" {
                let default = Array1::from_shape_vec(1, vec![0.0f32]).unwrap();
                let _ = replace_dataset(&sample, "thickness", &default);
            } else if name == "type" || name == "description" {
                let _ = clean_str_dataset::<256>(&sample, &name);
            }
        }
    }

    // list of source keys that fail in Mantid
    let source_keys = [
        "muon_energy",
        "muon_momentum",
        "muon_pulse_width",
        "name",
        "notes",
        "pion_momentum",
        "probe",
        "source_current",
        "source_energy",
        "source_frequency",
        "source_pulse_width",
        "target_material",
        "target_thickness",
        "type",
    ];

    let source = hist_data.group("instrument/source")?;
    for name in source.member_names()? {
        if source_keys.contains(&name.as_str()) {
            if let Ok(dataset) = source.dataset(&name) {
                let dtype = dataset.dtype()?;
                let desc = dtype.to_descriptor()?;
                match desc {
                    hdf5::types::TypeDescriptor::VarLenUnicode | hdf5::types::TypeDescriptor::FixedAscii(_) => {
                        let _ = clean_str_dataset::<100>(&source, &name);
                   }
                    _ => {println!("Unsupported dataset type: {:?}", desc);}
                }
            }
        }
   }

    Ok(())
}

pub fn get_p_info(file_name: &str) -> Result<(usize, usize)> {
    let file = File::open(file_name)?;
    let labels_ds = file.dataset("raw_data_1/periods/labels")?;
    let labels = if let Ok(lbl) = labels_ds.read_scalar::<VarLenUnicode>() {
        lbl.to_string()
    } else if let Ok(lbl) = labels_ds.read_scalar::<FixedAscii<256>>() {
        lbl.as_str().to_string()
    } else {
        return Err(anyhow!("Could not read labels dataset as string"));
    };
    let periods = labels.split(',').count();
    Ok((periods, 0)) // at present no Dwell info so its always zero 
}

pub fn save_default(
    output_file: &str,
    ref_file: &str,
    shapes: &HashMap<String, usize>,
) -> Result<()> {
    let file = File::open(ref_file)?;
    let new_file = File::open_rw(output_file)?;

    clean_up(&new_file)?;

    for key in file.member_names()? {
        let new_keys = new_file.member_names()?;
        let dest_group = if !new_keys.contains(&key) {
            new_file.create_group(&key)?
        } else {
            new_file.group(&key)?
        };

        let src_group = file.group(&key)?;
        
        let dest_group_keys = dest_group.member_names()?;
        for tmp_name in src_group.member_names()? {
            if tmp_name == "selog" {
                println!("skip");
            } else if src_group.group(&tmp_name.as_str()).is_ok() || src_group.dataset(&tmp_name.as_str()).is_ok() {
                // if a group
                 set_defaults(&src_group,
                  &dest_group,
                  tmp_name.as_str(),
                   shapes)?;
            }else{
                println!("not implemented for {}", tmp_name);
            }
        }
    }

    Ok(())
}
