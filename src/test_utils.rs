//! Utilities for testing.
//!
//! Note this file is only compiled when the package is compiled to run the test suite,
//! i.e. when `cargo test` is run.

use anyhow::Result;

use hdf5::{Dataset, File, Group, H5Type};
use ndarray::Array1;
use tempfile::NamedTempFile;

use crate::data::NexusData;

const FLOAT_EVENT_FIELDS: [&str; 3] = ["event_time_offset", "pulse_height", "event_time_zero"];
const INT_EVENT_FIELDS: [&str; 3] = ["event_id", "event_index", "period_number"];

/// we don't use `file` but we want it to continue to exist
#[allow(dead_code)]
pub struct MockData {
    file: NamedTempFile,
    nxs_file: File,
    pub event_data: Group,
    pub sample_logs: Group,
}

impl MockData {
    // Create a new empty mock data object.
    pub fn new() -> Result<MockData> {
        let tempfile = NamedTempFile::new()?;
        let file = File::create(tempfile.path())?;
        let data = file.create_group("raw_data_1")?;
        let event_data = data.create_group("detector_1_events")?;
        let sample_logs = data.create_group("selog")?;
        Ok(MockData {
            file: tempfile,
            nxs_file: file,
            event_data,
            sample_logs,
        })
    }

    /// Add a dataset to the mock data object.
    pub fn add_dataset<T>(&self, name: &str, data: Array1<T>) -> Result<Dataset>
    where
        T: H5Type,
    {
        let builder = self.event_data.new_dataset_builder();
        let dataset = builder.with_data(&data);
        Ok(dataset.create(name)?)
    }

    /// Turn the mock data object into a real NexusData object.
    pub fn create(&self, n_spec: usize, chunk_size: usize) -> Result<NexusData> {
        for field in FLOAT_EVENT_FIELDS {
            if self.event_data.dataset(field).is_err() {
                let dataset = self.event_data.new_dataset::<f64>();
                dataset.create(field)?;
            };
        }
        for field in INT_EVENT_FIELDS {
            if self.event_data.dataset(field).is_err() {
                let dataset = self.event_data.new_dataset::<i32>();
                dataset.create(field)?;
            };
        }
        let specs = self.event_data.dataset("event_id")?;
        let n_events = specs.size();
        let n_frames = self.event_data.dataset("period_number")?.size();
        Ok(NexusData {
            filename: "temp".to_string(),
            file: self.nxs_file.clone(),
            specs,
            times: self.event_data.dataset("event_time_offset")?,
            amps: self.event_data.dataset("pulse_height")?,
            frames: self.event_data.dataset("event_index")?,
            frame_times: self.event_data.dataset("event_time_zero")?,
            periods: self.event_data.dataset("period_number")?,
            sample_logs: self.sample_logs.clone(),
            sample_log_names: self.sample_logs.member_names()?,
            n_events,
            n_frames,
            n_spec,
            chunk_size,
        })
    }
}

/// Test creating a new MockData object.
#[test]
fn test_new() {
    let data = MockData::new();
    assert!(data.is_ok())
}

/// Test adding a dataset to the MockData object.
#[test]
fn test_add_dataset() {
    let data = MockData::new().unwrap();
    let vals = Array1::from_vec(vec![2., 3., 4.]);

    let _ = data.add_dataset("event_time_offset", vals.clone());
    let added_dataset = data.event_data.dataset("event_time_offset");
    assert!(added_dataset.is_ok());
    let array: Array1<f64> = added_dataset.unwrap().read_1d().unwrap();
    assert_eq!(array, vals)
}

/// Test that creating an empty NexusData object works.
#[test]
fn test_create_empty() {
    let data = MockData::new().unwrap();
    let create_result = data.create(64, 1);

    assert!(create_result.is_ok());

    // note empty arrays are not empty; they are stored as a scalar 0
    let nexus = create_result.unwrap();
    assert_eq!(nexus.specs.size(), 1);
    assert_eq!(nexus.times.size(), 1);
    assert_eq!(nexus.amps.size(), 1);
    assert_eq!(nexus.frames.size(), 1);
    assert_eq!(nexus.frame_times.size(), 1);
    assert_eq!(nexus.periods.size(), 1);
}

/// Test that creating an empty NexusData with an array predefined works.
#[test]
fn test_create_with_arrays() {
    let data = MockData::new().unwrap();

    let vals = Array1::from_vec(vec![2., 3., 4.]);
    let _ = data.add_dataset("event_time_offset", vals.clone());

    let create_result = data.create(64, 1);
    assert!(create_result.is_ok());

    // note empty arrays are not empty; they are stored as a scalar 0
    let nexus = create_result.unwrap();
    assert_eq!(nexus.specs.size(), 1);
    assert_eq!(nexus.times.read_1d::<f64>().unwrap(), vals);
    assert_eq!(nexus.amps.size(), 1);
    assert_eq!(nexus.frames.size(), 1);
    assert_eq!(nexus.frame_times.size(), 1);
    assert_eq!(nexus.periods.size(), 1);
}
