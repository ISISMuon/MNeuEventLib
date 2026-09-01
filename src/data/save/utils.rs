use std::str::FromStr;

use anyhow::Result;
use hdf5::types::{FixedAscii, H5Type, VarLenUnicode};
use hdf5::{Dataset, File, Group, Location};
use ndarray::{arr0, Array, Array1, Dimension};


// placeholder type for data that will just be copied into the nexus file
pub type CopyData = ();

pub trait SaveFile {
    /// Save an object to file.
    fn save(&self, filename: String, event_data: &File) -> Result<()>;
}

pub trait Save {
    /// Save an object to a HDF5 group.
    fn save(&self, group: &Group, event_data: &Group) -> Result<()>;
}

/// Create a dataset with a scalar in it.
pub fn add_scalar<T: H5Type>(group: &Group, scalar: T, name: &str) -> Result<Dataset> {
    // note the 'scalars' in histogram files are actually just length-1 vectors
    let data: Array1<T> = Array1::from_vec(vec![scalar]);
    add_array(group, &data, name)
}

/// Create a dataset with an array in it and return the dataset.
pub fn add_array<T: H5Type, D: Dimension>(
    group: &Group,
    array: &Array<T, D>,
    name: &str,
) -> Result<Dataset> {
    let builder = group.new_dataset_builder();
    let builder = builder.with_data(array);
    builder.create(name)?;
    Ok(group.dataset(name)?)
}

pub fn add_str_scalar<const LEN: usize>(
    group: &Group,
    scalar: &str,
    name: &str,
) -> Result<Dataset> {
    let string = FixedAscii::<LEN>::from_ascii(scalar)?;
    add_scalar(group, string, name)
}

pub fn copy_scalar<T: H5Type + Clone>(from: &Group, to: &Group, item: &str) -> Result<Dataset> {
    let dataset = from.dataset(item)?.read_1d::<T>();
    // some 'scalars' are 1D arrays with 1 element, so we need to handle it...
    let data: T = if let Ok(array) = dataset {
        array[0].clone()
    } else {
        from.dataset(item)?.read()?.into_scalar()
    };
    add_scalar(to, data, item)
}

/// Set an attribute of a group.
pub fn add_attr<T: H5Type>(loc: &Location, data: T, name: &str) -> Result<()> {
    let builder = loc.new_attr_builder();
    let scalar = arr0(data);
    let builder = builder.with_data(&scalar);
    builder.create(name)?;
    Ok(())
}

pub fn add_str_attr<const LEN: usize>(loc: &Location, data: &str, name: &str) -> Result<()> {
    let string = FixedAscii::<LEN>::from_ascii(data)?;
    add_attr(loc, string, name)
}

/// Set the NX_class of a group.
pub fn add_nx_class(group: &Group, class: &str) -> Result<()> {
    let string = VarLenUnicode::from_str(class)?;
    add_attr(group, string, "NX_class")
}

#[cfg(test)]
mod tests {
    use super::*;
    use hdf5::File;
    use ndarray::Array2;
    use std::env::temp_dir;
    use std::str::FromStr;

    fn tmp_file(name: &str) -> File {
        let mut tmp_path = temp_dir();
        tmp_path.push(format!("utils_test_{name}.nxs"));
        File::create(tmp_path).unwrap()
    }

    /// `add_array` should work for multi-dimensional arrays too.
    #[test]
    fn test_add_array() {
        let file = tmp_file("add_array_2d");
        let group = file.create_group("g").unwrap();
        let data: Array2<f32> =
            Array2::from_shape_vec((2, 3), vec![1., 2., 3., 4., 5., 6.]).unwrap();

        let dataset = add_array(&group, &data, "matrix").unwrap();
        let read: Array2<f32> = dataset.read_2d().unwrap();
        assert_eq!(read, data);
        assert_eq!(dataset.shape(), vec![2, 3]);
    }

    /// `add_scalar` should create a length-1 dataset containing the scalar,
    /// readable back out as a scalar.
    #[test]
    fn test_add_scalar() {
        let file = tmp_file("add_scalar");
        let group = file.create_group("g").unwrap();

        add_scalar(&group, 42i32, "answer").unwrap();

        let value: i32 = group.dataset("answer").unwrap().read_1d().unwrap()[0];
        assert_eq!(value, 42);
    }

    /// `add_scalar` should work for floating point scalars too.
    #[test]
    fn test_add_scalar_float() {
        let file = tmp_file("add_scalar_float");
        let group = file.create_group("g").unwrap();

        add_scalar(&group, 3.1, "scalar").unwrap();

        let value: f32 = group.dataset("scalar").unwrap().read_1d().unwrap()[0];
        assert_eq!(value, 3.1);
    }

    /// `add_str_scalar` should store a fixed-length ASCII string that reads
    /// back correctly.
    #[test]
    fn test_add_str_scalar() {
        let file = tmp_file("add_str_scalar");
        let group = file.create_group("g").unwrap();

        add_str_scalar::<8>(&group, "hello", "greeting").unwrap();

        let value: hdf5::types::FixedAscii<8> =
            group.dataset("greeting").unwrap().read_1d().unwrap()[0];
        assert_eq!(value.as_str(), "hello");
    }

    /// `add_str_scalar` should error if the string is longer than the fixed length.
    #[test]
    fn test_add_str_scalar_too_long_errors() {
        let file = tmp_file("add_str_scalar_too_long");
        let group = file.create_group("g").unwrap();

        let result = add_str_scalar::<3>(&group, "toolong", "field");
        assert!(result.is_err());
    }

    /// `copy_scalar` should read a scalar of type T from one group's dataset
    /// and write an equivalent dataset into another group.
    #[test]
    fn test_copy_scalar() {
        let file = tmp_file("copy_scalar");
        let source = file.create_group("source").unwrap();
        let dest = file.create_group("dest").unwrap();

        // the scalars in event files are actually scalars, but add_scalar adds a 1d array
        // so we need to do this differently here
        add_array(&source, &arr0(99i32), "count").unwrap();

        copy_scalar::<i32>(&source, &dest, "count").unwrap();

        let value: i32 = dest.dataset("count").unwrap().read_1d().unwrap()[0];
        assert_eq!(value, 99);
    }

    /// `copy_scalar` should work with string (VarLenUnicode) types.
    #[test]
    fn test_copy_scalar_string() {
        let file = tmp_file("copy_scalar_string");
        let source = file.create_group("source").unwrap();
        let dest = file.create_group("dest").unwrap();

        let text = hdf5::types::VarLenUnicode::from_str("hello world").unwrap();
        // the scalars in event files are actually scalars, but add_scalar adds a 1d array
        // so we need to do this differently here
        add_array(&source, &arr0(text), "title").unwrap();

        copy_scalar::<hdf5::types::VarLenUnicode>(&source, &dest, "title").unwrap();

        let value: &hdf5::types::VarLenUnicode =
            &dest.dataset("title").unwrap().read_1d().unwrap()[0];
        assert_eq!(value.as_str(), "hello world");
    }

    /// `add_attr` should set a scalar attribute on a group that is readable
    /// back out.
    #[test]
    fn test_add_attr() {
        let file = tmp_file("add_attr");
        let group = file.create_group("g").unwrap();

        add_attr(&group, 7i32, "my_attr").unwrap();

        let value: i32 = group.attr("my_attr").unwrap().read_scalar().unwrap();
        assert_eq!(value, 7);
    }

    /// `add_str_attr` should set a fixed-length ASCII string attribute.
    #[test]
    fn test_add_str_attr() {
        let file = tmp_file("add_str_attr");
        let group = file.create_group("g").unwrap();

        add_str_attr::<5>(&group, "units", "label").unwrap();

        let value: hdf5::types::FixedAscii<5> = group.attr("label").unwrap().read_scalar().unwrap();
        assert_eq!(value.as_str(), "units");
    }

    /// `add_nx_class` should set the `NX_class` attribute to the given class name.
    #[test]
    fn test_add_nx_class() {
        let file = tmp_file("add_nx_class");
        let group = file.create_group("g").unwrap();

        add_nx_class(&group, "NXinstrument").unwrap();

        let value: hdf5::types::VarLenUnicode =
            group.attr("NX_class").unwrap().read_scalar().unwrap();
        assert_eq!(value.as_str(), "NXinstrument");
    }
}
