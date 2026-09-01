use std::collections::HashMap;
use std::str::FromStr;

use anyhow::{anyhow, Result};
use hdf5::types::{FixedAscii, VarLenUnicode};
use hdf5::{Dataset, File, Group, Location};
use ndarray::Array1;
use crate::data::save::sanitise::utils::*;
use crate::data::save::utils::*;


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
