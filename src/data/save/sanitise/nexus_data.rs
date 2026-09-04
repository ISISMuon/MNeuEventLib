use std::collections::HashMap;

use crate::data::save::sanitise::utils::*;
use crate::data::save::utils::*;
use anyhow::{anyhow, Result};
use hdf5::types::{FixedAscii, VarLenUnicode};
use hdf5::{File, Group};
use ndarray::Array1;

/// Create a new dataset in the destination file with default values.
/// These values are given in the reference file (made from tools/make_default.py
/// and the length of the data array depends on the number of periods in the data)
///
/// Parameters
/// ----------
/// * `default` - Reference group containing default values
/// * `dest` - Destination group where default values will be added
/// * `name` - Name of the dataset to create
/// * `shapes` - Hashmap which defines the length of each dataset (in terms of periods)
///
/// Returns
/// -------
/// * `Ok(())` - If the dataset is created successfully
/// * `Err(anyhow::Error)` - If the dataset cannot be created
fn create_default_dataset(
    default: &Group,
    dest: &Group,
    _name: &str,
    shapes: &HashMap<String, usize>,
) -> Result<()> {
    let name = default.name().split("dataset_").last().unwrap().to_string();
    let key: String = default
        .dataset("shape")
        .unwrap()
        .read_scalar::<hdf5::types::VarLenUnicode>()
        .unwrap()
        .as_str()
        .to_string();
    let len: usize = *shapes.get(&key).unwrap();
    let dtype: &hdf5::types::VarLenUnicode =
        &default.dataset("dtype").unwrap().read_scalar().unwrap();
    if dtype.as_str() == "int32" {
        let default_value = default.dataset("default").unwrap().read_scalar::<i32>()?;
        if let Err(e) = add_array(dest, &Array1::from_elem(len, default_value), &name) {
            eprintln!("error: {e}");
            eprintln!("chain: {e:?}");
        };
    } else if dtype.as_str() == "float32" {
        let default_value = default.dataset("default").unwrap().read_scalar::<f32>()?;
        add_array(dest, &Array1::from_elem(len, default_value), &name)?;
    } else if dtype.as_str() == "float64" {
        let default_value = default.dataset("default").unwrap().read_scalar::<f64>()?;
        add_array(dest, &Array1::from_elem(len, default_value), &name)?;
    } else {
        return Err(anyhow!("Unsupported default dataset type {:?}", dtype));
    };
    for att_name in default.attr_names()? {
        let data = dest.dataset(name.as_str())?;
        println!("copy attribute {}", att_name);
        copy_attr(default, &data, &att_name)?;
    }
    Ok(())
}

/// Adds the missing data to the destination file based on the source (reference)
/// file and the shape information.
///
/// ## Arguments
/// * `source_parent` - The parent group  from the source (reference) data
/// * `dest` - The parent group from the destination data
/// * `name` - The name of the dataset or group to add
/// * `shapes` - Hashmap which defines the length of each dataset (in terms of periods)
///
/// ## Returns
/// * `Ok(())` - If the dataset or group is added successfully
/// * `Err(anyhow::Error)` - If the dataset or group cannot be added
fn set_defaults(
    source_parent: &Group,
    dest: &Group,
    name: &str,
    shapes: &HashMap<String, usize>,
) -> Result<()> {
    if name.contains("dataset_") && dest.dataset(name.replace("dataset_", "").as_str()).is_err() {
        /* we know this is a group that defines a period
        dependent dataset */
        println!("create default {}", name);
        create_default_dataset(&source_parent.group(name).unwrap(), dest, name, shapes)?;
        Ok(())
    } else if source_parent.dataset(name).is_ok() && dest.dataset(name).is_ok() {
        // if dataset exists in both files
        println!("{} dataset already exists", name);
        let src_ds = source_parent.dataset(name)?;
        let dst_ds = dest.dataset(name)?;
        for att_name in src_ds.attr_names()? {
            println!("copy attribute {}", att_name);
            copy_attr(&src_ds, &dst_ds, &att_name)?;
        }
        Ok(())
    } else if source_parent.dataset(name).is_ok() && dest.dataset(name).is_err() {
        // if dataset exists in source but not in destination
        println!("copy dataset {}", name);
        source_parent.dataset(name)?.copy_to(dest, name)?;
        Ok(())
    } else if source_parent.group(name).is_ok() && dest.group(name).is_ok() {
        // if group exists in both files
        println!("group {} exists in both files, going deeper", name);
        for member in source_parent.group(name).unwrap().member_names()? {
            set_defaults(
                &source_parent.group(name).unwrap(),
                &dest.group(name).unwrap(),
                member.as_str(),
                shapes,
            )?;
        }
        Ok(())
    } else if source_parent.group(name).is_ok() && dest.group(name).is_err() {
        // copy group
        println!("make a copy of group {}", name);
        /* this works for muons as non of the datasets from the missing group have a
        length that depends on the number of periods. */
        source_parent.group(name)?.copy_to(dest, name)?;
        Ok(())
    } else {
        Err(anyhow!("{} does not exist", name))
    }
}

/// Replaces "broken" datasets with "fixed" ones (numbers are recorded as null UTF8's)
///
/// ## Arguments
/// * `new_file` - The destination file
///
/// ## Returns
/// * `Ok(())` - If the datasets are removed successfully
/// * `Err(anyhow::Error)` - If the datasets cannot be removed
fn clean_up(new_file: &File) -> Result<()> {
    let hist_data = new_file.group("raw_data_1")?;

    let _ = replace_str_dataset::<4>(&hist_data, "name", "HIFI", "name");
    let _ = replace_str_dataset::<4>(&hist_data, "title", "Data", "");

    if let Ok(sample) = hist_data.group("sample") {
        for name in sample.member_names()? {
            if let Ok(_group) = sample.group(&name) {
                continue;
            } else if let Ok(_dataset) = sample.dataset(&name) {
                if name == "thickness" {
                    let default = Array1::from_shape_vec(1, vec![0.0f32]).unwrap();
                    let _ = replace_dataset(&sample, "thickness", &default);
                } else if name == "type" || name == "description" {
                    let _ = clean_str_dataset::<256>(&sample, &name);
                }
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

    if let Ok(source) = hist_data.group("instrument/source") {
        for name in source.member_names()? {
            if source_keys.contains(&name.as_str()) {
                if let Ok(dataset) = source.dataset(&name) {
                    let dtype = dataset.dtype()?;
                    let desc = dtype.to_descriptor()?;
                    match desc {
                        hdf5::types::TypeDescriptor::VarLenUnicode
                        | hdf5::types::TypeDescriptor::FixedAscii(_) => {
                            let _ = clean_str_dataset::<100>(&source, &name);
                        }
                        _ => {
                            println!("Unsupported dataset type: {:?}", desc);
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Gets the number of periods and Dwell for the given file.
///
/// ## Arguments
/// * `file_name` - The name of the file to get the period information from
///
/// ## Returns
/// * `Ok((periods, dwell))` - The number of periods and Dwell
/// * `Err(anyhow::Error)` - If the period information cannot be retrieved
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

/// Saves default values to the output file based on the reference file and shape information.
///
/// ## Arguments
/// * `output_file` - The name of the output file to save the default values to
/// * `ref_file` - The name of the reference file to get the default values from
/// * `shapes` - Hashmap which defines the length of each dataset (in terms of periods)
///
/// ## Returns
/// * `Ok(())` - If the default values are saved successfully
/// * `Err(anyhow::Error)` - If the default values cannot be saved
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

        let _dest_group_keys = dest_group.member_names()?;
        for tmp_name in src_group.member_names()? {
            if tmp_name == "selog" {
                println!("skip");
            } else if src_group.group(tmp_name.as_str()).is_ok()
                || src_group.dataset(tmp_name.as_str()).is_ok()
            {
                // if a group
                set_defaults(&src_group, &dest_group, tmp_name.as_str(), shapes)?;
            } else {
                println!("not implemented for {}", tmp_name);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hdf5::File;
    use ndarray::{arr0, Array1};
    use std::str::FromStr;
    use tempfile::tempdir;

    fn create_test_file(
        name: &str,
    ) -> (tempfile::TempDir, File, std::sync::MutexGuard<'static, ()>) {
        let guard = crate::test_utils::lock_hdf5_test();
        let dir = tempdir().unwrap();
        let path = dir.path().join(format!("{name}.nxs"));
        let file = File::create(&path).unwrap();
        (dir, file, guard)
    }

    #[test]
    fn test_get_p_info_varlen_unicode() {
        let (dir, file, _guard) = create_test_file("test_get_p_info_varlen");
        let path = dir.path().join("test_get_p_info_varlen.nxs");

        let raw = file.create_group("raw_data_1").unwrap();
        let periods_grp = raw.create_group("periods").unwrap();
        let labels = VarLenUnicode::from_str("Period1,Period2,Period3").unwrap();
        add_array(&periods_grp, &arr0(labels), "labels").unwrap();
        drop(file);

        let (periods, dwell) = get_p_info(path.to_str().unwrap()).unwrap();
        assert_eq!(periods, 3);
        assert_eq!(dwell, 0);
    }

    #[test]
    fn test_get_p_info_fixed_ascii() {
        let (dir, file, _guard) = create_test_file("test_get_p_info_fixed");
        let path = dir.path().join("test_get_p_info_fixed.nxs");

        let raw = file.create_group("raw_data_1").unwrap();
        let periods_grp = raw.create_group("periods").unwrap();
        let labels = FixedAscii::<256>::from_ascii(b"P1,P2,P3,P4,P5").unwrap();
        add_array(&periods_grp, &arr0(labels), "labels").unwrap();
        drop(file);
        let (periods, dwell) = get_p_info(path.to_str().unwrap()).unwrap();
        assert_eq!(periods, 5);
        assert_eq!(dwell, 0);
    }

    #[test]
    fn test_get_p_info_nonexistent_file() {
        let res = get_p_info("nonexistent_path_file.nxs");
        assert!(res.is_err());
    }

    #[test]
    fn test_get_p_info_missing_labels() {
        let (dir, file, _guard) = create_test_file("test_get_p_info_missing_labels");
        let path = dir.path().join("test_get_p_info_missing_labels.nxs");
        file.create_group("raw_data_1").unwrap();
        drop(file);

        let res = get_p_info(path.to_str().unwrap());
        assert!(res.is_err());
    }

    #[test]
    fn test_create_default_dataset_int32() {
        let (_dir, file, _guard) = create_test_file("test_create_default_int32");
        let default_grp = file.create_group("raw_data_1/dataset_counts").unwrap();
        let dest_grp = file.create_group("dest").unwrap();

        let shape_str = VarLenUnicode::from_str("nperiods").unwrap();
        add_array(&default_grp, &arr0(shape_str), "shape").unwrap();

        let dtype_str = VarLenUnicode::from_str("int32").unwrap();
        add_array(&default_grp, &arr0(dtype_str), "dtype").unwrap();

        add_array(&default_grp, &arr0(42i32), "default").unwrap();
        add_str_attr::<5>(&default_grp, "units", "unit").unwrap();

        let mut shapes = HashMap::new();
        shapes.insert("nperiods".to_string(), 4);

        create_default_dataset(&default_grp, &dest_grp, "dataset_counts", &shapes).unwrap();

        let created_ds = dest_grp.dataset("counts").unwrap();
        let vals: Array1<i32> = created_ds.read_1d().unwrap();
        assert_eq!(vals.to_vec(), vec![42, 42, 42, 42]);

        let attr_val: FixedAscii<5> = created_ds.attr("unit").unwrap().read_scalar().unwrap();
        assert_eq!(attr_val.as_str(), "units");
    }

    #[test]
    fn test_create_default_dataset_float32() {
        let (_dir, file, _guard) = create_test_file("test_create_default_float32");
        let default_grp = file.create_group("raw_data_1/dataset_temp").unwrap();
        let dest_grp = file.create_group("dest").unwrap();

        let shape_str = VarLenUnicode::from_str("nperiods").unwrap();
        add_array(&default_grp, &arr0(shape_str), "shape").unwrap();

        let dtype_str = VarLenUnicode::from_str("float32").unwrap();
        add_array(&default_grp, &arr0(dtype_str), "dtype").unwrap();

        add_array(&default_grp, &arr0(1.5f32), "default").unwrap();

        let mut shapes = HashMap::new();
        shapes.insert("nperiods".to_string(), 2);

        create_default_dataset(&default_grp, &dest_grp, "dataset_temp", &shapes).unwrap();

        let created_ds = dest_grp.dataset("temp").unwrap();
        let vals: Array1<f32> = created_ds.read_1d().unwrap();
        assert_eq!(vals.to_vec(), vec![1.5, 1.5]);
    }

    #[test]
    fn test_create_default_dataset_float64() {
        let (_dir, file, _guard) = create_test_file("test_create_default_float64");
        let default_grp = file.create_group("raw_data_1/dataset_time").unwrap();
        let dest_grp = file.create_group("dest").unwrap();

        let shape_str = VarLenUnicode::from_str("single").unwrap();
        add_array(&default_grp, &arr0(shape_str), "shape").unwrap();

        let dtype_str = VarLenUnicode::from_str("float64").unwrap();
        add_array(&default_grp, &arr0(dtype_str), "dtype").unwrap();

        add_array(&default_grp, &arr0(4.14f64), "default").unwrap();

        let mut shapes = HashMap::new();
        shapes.insert("single".to_string(), 1);

        create_default_dataset(&default_grp, &dest_grp, "dataset_time", &shapes).unwrap();

        let created_ds = dest_grp.dataset("time").unwrap();
        let vals: Array1<f64> = created_ds.read_1d().unwrap();
        assert_eq!(vals.to_vec(), vec![4.14]);
    }

    #[test]
    fn test_create_default_dataset_unsupported_dtype() {
        let (_dir, file, _guard) = create_test_file("test_create_default_unsupported");
        let default_grp = file.create_group("raw_data_1/dataset_bad").unwrap();
        let dest_grp = file.create_group("dest").unwrap();

        let shape_str = VarLenUnicode::from_str("nperiods").unwrap();
        add_array(&default_grp, &arr0(shape_str), "shape").unwrap();

        let dtype_str = VarLenUnicode::from_str("unsupported_type").unwrap();
        add_array(&default_grp, &arr0(dtype_str), "dtype").unwrap();

        let mut shapes = HashMap::new();
        shapes.insert("nperiods".to_string(), 1);

        let res = create_default_dataset(&default_grp, &dest_grp, "dataset_bad", &shapes);
        assert!(res.is_err());
    }

    #[test]
    fn test_set_defaults_dataset_prefixed_default() {
        let (_dir, file, _guard) = create_test_file("test_set_defaults_prefixed");
        let src_parent = file.create_group("src").unwrap();
        let dest_parent = file.create_group("dest").unwrap();

        let default_grp = src_parent.create_group("dataset_rate").unwrap();
        let shape_str = VarLenUnicode::from_str("nperiods").unwrap();
        add_array(&default_grp, &arr0(shape_str), "shape").unwrap();
        let dtype_str = VarLenUnicode::from_str("int32").unwrap();
        add_array(&default_grp, &arr0(dtype_str), "dtype").unwrap();
        add_array(&default_grp, &arr0(10i32), "default").unwrap();

        let mut shapes = HashMap::new();
        shapes.insert("nperiods".to_string(), 3);

        set_defaults(&src_parent, &dest_parent, "dataset_rate", &shapes).unwrap();

        let created_ds = dest_parent.dataset("rate").unwrap();
        let vals: Array1<i32> = created_ds.read_1d().unwrap();
        assert_eq!(vals.to_vec(), vec![10, 10, 10]);
    }

    #[test]
    fn test_set_defaults_dataset_already_exists() {
        let (_dir, file, _guard) = create_test_file("test_set_defaults_already_exists");
        let src_parent = file.create_group("src").unwrap();
        let dest_parent = file.create_group("dest").unwrap();

        let default_grp = src_parent.create_group("dataset_rate").unwrap();
        let shape_str = VarLenUnicode::from_str("nperiods").unwrap();
        add_array(&default_grp, &arr0(shape_str), "shape").unwrap();
        let dtype_str = VarLenUnicode::from_str("int32").unwrap();
        add_array(&default_grp, &arr0(dtype_str), "dtype").unwrap();
        add_array(&default_grp, &arr0(10i32), "default").unwrap();

        let vals = Array1::from_vec(vec![1i32, 3i32, 6i32]);
        add_array(&dest_parent, &vals, "rate").unwrap();

        let mut shapes = HashMap::new();
        shapes.insert("nperiods".to_string(), 3);

        set_defaults(&src_parent, &dest_parent, "dataset_rate", &shapes).unwrap();

        let created_ds = dest_parent.dataset("rate").unwrap();
        let vals: Array1<i32> = created_ds.read_1d().unwrap();
        assert_eq!(vals.to_vec(), vec![1, 3, 6]);
    }

    #[test]
    fn test_set_defaults_dataset_copy_when_missing_in_dest() {
        let (_dir, file, _guard) = create_test_file("test_set_defaults_copy_ds");
        let src_parent = file.create_group("src").unwrap();
        let dest_parent = file.create_group("dest").unwrap();

        let data = Array1::from_vec(vec![1.0f32, 2.0f32]);
        add_array(&src_parent, &data, "wavelength").unwrap();

        let shapes = HashMap::new();
        set_defaults(&src_parent, &dest_parent, "wavelength", &shapes).unwrap();

        let copied_ds = dest_parent.dataset("wavelength").unwrap();
        let vals: Array1<f32> = copied_ds.read_1d().unwrap();
        assert_eq!(vals.to_vec(), vec![1.0, 2.0]);
    }

    #[test]
    fn test_set_defaults_group_copy_when_missing_in_dest() {
        let (_dir, file, _guard) = create_test_file("test_set_defaults_copy_group");
        let src_parent = file.create_group("src").unwrap();
        let dest_parent = file.create_group("dest").unwrap();

        let src_sub = src_parent.create_group("extra_group").unwrap();
        let data = Array1::from_vec(vec![100i32]);
        add_array(&src_sub, &data, "item").unwrap();

        let shapes = HashMap::new();
        set_defaults(&src_parent, &dest_parent, "extra_group", &shapes).unwrap();

        let copied_sub = dest_parent.group("extra_group").unwrap();
        let vals: Array1<i32> = copied_sub.dataset("item").unwrap().read_1d().unwrap();
        assert_eq!(vals.to_vec(), vec![100]);
    }

    #[test]
    fn test_set_defaults_group_exists_in_both() {
        let (_dir, file, _guard) = create_test_file("test_set_defaults_group_both");
        let src_parent = file.create_group("src").unwrap();
        let dest_parent = file.create_group("dest").unwrap();

        let src_sub = src_parent.create_group("common_group").unwrap();
        let _ = dest_parent.create_group("common_group").unwrap();

        let data = Array1::from_vec(vec![5i32]);
        add_array(&src_sub, &data, "nested_data").unwrap();

        let shapes = HashMap::new();
        set_defaults(&src_parent, &dest_parent, "common_group", &shapes).unwrap();

        let dest_sub = dest_parent.group("common_group").unwrap();
        let vals: Array1<i32> = dest_sub.dataset("nested_data").unwrap().read_1d().unwrap();
        assert_eq!(vals.to_vec(), vec![5]);
    }

    #[test]
    fn test_set_defaults_nonexistent_returns_err() {
        let (_dir, file, _guard) = create_test_file("test_set_defaults_nonexistent");
        let src_parent = file.create_group("src").unwrap();
        let dest_parent = file.create_group("dest").unwrap();

        let shapes = HashMap::new();
        let res = set_defaults(&src_parent, &dest_parent, "missing_item", &shapes);
        assert!(res.is_err());
    }

    #[test]
    fn test_clean_up() {
        let (_dir, file, _guard) = create_test_file("test_clean_up");
        let raw = file.create_group("raw_data_1").unwrap();

        let name_str = VarLenUnicode::from_str("name").unwrap();
        add_array(&raw, &arr0(name_str), "name").unwrap();

        let title_str = VarLenUnicode::from_str("").unwrap();
        add_array(&raw, &arr0(title_str), "title").unwrap();

        let sample = raw.create_group("sample").unwrap();
        let thickness_data = Array1::from_vec(vec![12.5f32]);
        add_array(&sample, &thickness_data, "thickness").unwrap();
        let sample_type_str = VarLenUnicode::from_str("").unwrap();
        add_array(&sample, &arr0(sample_type_str), "type").unwrap();
        let sample_desc_str = VarLenUnicode::from_str("Sample1").unwrap();
        add_array(&sample, &arr0(sample_desc_str), "description").unwrap();

        let source = raw.create_group("instrument/source").unwrap();
        let muon_energy_str = VarLenUnicode::from_str("").unwrap();
        add_array(&source, &arr0(muon_energy_str), "muon_energy").unwrap();
        let target_mat_str = VarLenUnicode::from_str("Carbon").unwrap();
        add_array(&source, &arr0(target_mat_str), "target_material").unwrap();

        clean_up(&file).unwrap();

        let new_name: FixedAscii<4> = raw.dataset("name").unwrap().read_1d().unwrap()[0];
        assert_eq!(new_name.as_str(), "HIFI");

        let new_title: FixedAscii<4> = raw.dataset("title").unwrap().read_1d().unwrap()[0];
        assert_eq!(new_title.as_str(), "Data");

        let new_thickness: Array1<f32> = sample.dataset("thickness").unwrap().read_1d().unwrap();
        assert_eq!(new_thickness.to_vec(), vec![0.0f32]);

        let new_type: FixedAscii<7> = sample.dataset("type").unwrap().read_1d().unwrap()[0];
        assert_eq!(new_type.as_str(), "Missing");

        let new_desc: FixedAscii<256> =
            sample.dataset("description").unwrap().read_1d().unwrap()[0];
        assert_eq!(new_desc.as_str(), "Sample1");

        let new_muon_energy: FixedAscii<7> =
            source.dataset("muon_energy").unwrap().read_1d().unwrap()[0];
        assert_eq!(new_muon_energy.as_str(), "Missing");

        let new_target_mat: FixedAscii<100> = source
            .dataset("target_material")
            .unwrap()
            .read_1d()
            .unwrap()[0];
        assert_eq!(new_target_mat.as_str(), "Carbon");
    }

    #[test]
    fn test_save_default_integration() {
        let _guard = crate::test_utils::lock_hdf5_test();
        let dir = tempdir().unwrap();
        let ref_path = dir.path().join("ref.nxs");
        let out_path = dir.path().join("out.nxs");

        // Create reference file
        {
            let ref_file = File::create(&ref_path).unwrap();
            let ref_raw = ref_file.create_group("raw_data_1").unwrap();

            let def_grp = ref_raw.create_group("dataset_beam_counts").unwrap();
            let shape_str = VarLenUnicode::from_str("nperiods").unwrap();
            add_array(&def_grp, &arr0(shape_str), "shape").unwrap();
            let dtype_str = VarLenUnicode::from_str("int32").unwrap();
            add_array(&def_grp, &arr0(dtype_str), "dtype").unwrap();
            add_array(&def_grp, &arr0(55i32), "default").unwrap();

            let ref_inst = ref_raw.create_group("instrument_ref").unwrap();
            let inst_data = Array1::from_vec(vec![1.23f32]);
            add_array(&ref_inst, &inst_data, "setting").unwrap();
        }

        // Create output file
        {
            let out_file = File::create(&out_path).unwrap();
            let out_raw = out_file.create_group("raw_data_1").unwrap();

            let name_str = VarLenUnicode::from_str("name").unwrap();
            add_array(&out_raw, &arr0(name_str), "name").unwrap();

            let title_str = VarLenUnicode::from_str("").unwrap();
            add_array(&out_raw, &arr0(title_str), "title").unwrap();

            let sample = out_raw.create_group("sample").unwrap();
            let thickness_data = Array1::from_vec(vec![9.0f32]);
            add_array(&sample, &thickness_data, "thickness").unwrap();

            let _source = out_raw.create_group("instrument/source").unwrap();
        }

        let mut shapes = HashMap::new();
        shapes.insert("nperiods".to_string(), 2);

        save_default(
            out_path.to_str().unwrap(),
            ref_path.to_str().unwrap(),
            &shapes,
        )
        .unwrap();

        // Verify output file
        let out_file = File::open(&out_path).unwrap();
        let out_raw = out_file.group("raw_data_1").unwrap();

        // Check cleaned values
        let name_val: FixedAscii<4> = out_raw.dataset("name").unwrap().read_1d().unwrap()[0];
        assert_eq!(name_val.as_str(), "HIFI");

        // Check default dataset was created
        let beam_counts: Array1<i32> = out_raw.dataset("beam_counts").unwrap().read_1d().unwrap();
        assert_eq!(beam_counts.to_vec(), vec![55, 55]);

        // Check group was copied from ref
        let inst_setting: Array1<f32> = out_raw
            .dataset("instrument_ref/setting")
            .unwrap()
            .read_1d()
            .unwrap();
        assert_eq!(inst_setting.to_vec(), vec![1.23]);
    }
}
