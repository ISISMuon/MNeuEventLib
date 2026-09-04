use std::str::FromStr;

use anyhow::{Error, Result};
use hdf5::types::{H5Type, VarLenUnicode};
use hdf5::Group;
use ndarray::Array1;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::data::save::utils::*;
use crate::data::{NexusData, SampleLog, ValueLog};
use crate::filters::Filters;
use crate::utils::NarrowTo32;

/// Save the log to a HDF5 Group
impl Save for SampleLog {
    fn save(&self, group: &Group, _: &Group) -> Result<()> {
        match self {
            SampleLog::I8(log) => log.save_to_group(group),
            SampleLog::I16(log) => log.save_to_group(group),
            SampleLog::I32(log) => log.save_to_group(group),
            SampleLog::I64(log) => log.save_with_narrowing(group),
            SampleLog::U8(log) => log.save_to_group(group),
            SampleLog::U16(log) => log.save_to_group(group),
            SampleLog::U32(log) => log.save_to_group(group),
            SampleLog::U64(log) => log.save_with_narrowing(group),
            SampleLog::F32(log) => log.save_to_group(group),
            SampleLog::F64(log) => log.save_with_narrowing(group),
        }
    }
}

impl<T> ValueLog<T>
where
    T: NarrowTo32 + Copy,
{
    /// Save a log to a group, reducing entry size to 32-bit.
    pub fn save_with_narrowing(&self, group: &Group) -> Result<()> {
        let log_group = group.create_group(&self.name)?;
        let value_log = log_group.create_group("value_log")?;
        let times: Array1<f32> = self.time.map(|t| *t as f32);
        let values = self.value.map(|v| v.narrow());
        let time_dataset = add_array(&value_log, &times, "time")?;
        add_str_attr::<7>(&time_dataset, "seconds", "units")?;
        let value_dataset = add_array(&value_log, &values, "value")?;
        add_attr(
            &value_dataset,
            VarLenUnicode::from_str(&self.unit)?,
            "units",
        )?;
        Ok(())
    }
}

impl<T> ValueLog<T>
where
    T: H5Type,
{
    /// Save a log to a group.
    pub fn save_to_group(&self, group: &Group) -> Result<()> {
        let log_group = group.create_group(&self.name)?;
        let value_log = log_group.create_group("value_log")?;
        let times: Array1<f32> = self.time.map(|t| *t as f32);
        let time_dataset = add_array(&value_log, &times, "time")?;
        add_str_attr::<7>(&time_dataset, "seconds", "units")?;
        let value_dataset = add_array(&value_log, &self.value, "value")?;
        add_attr(
            &value_dataset,
            VarLenUnicode::from_str(&self.unit)?,
            "units",
        )?;
        Ok(())
    }
}

pub fn get_all_sample_logs(event_data: &NexusData, filters: &Filters) -> Result<Vec<SampleLog>> {
    let (mut time_starts, mut time_ends) = filters.get_time_filter_times();

    let log_names = filters.get_required_log_names();

    let value_logs = match event_data.get_sample_logs(log_names) {
        Ok(logs) => logs,
        Err(info) => return Err(Error::msg(format!("Failed to get logs: {info}"))),
    };
    let (log_starts, log_ends) = filters.get_log_filter_times(value_logs);

    time_starts.extend(log_starts);
    time_ends.extend(log_ends);

    Ok(event_data
        .sample_log_names
        .par_iter()
        // we use filter_map to skip unloadable sample logs
        .filter_map(|name| {
            match event_data.get_sample_log(name) {
                Ok(sample_log) => match time_starts.is_empty() {
                    true => Some(sample_log),
                    false => Some(sample_log.apply_filters(&time_starts, &time_ends)),
                }
                Err(error) => {
                    // if sample log is unsupported, ignore in output
                    println!("Sample log {name} failed to load: ignoring in output file.\nError: {error}");
                    None
                }
        }})
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch_interface::FilterIndex;
    use crate::BatchData;
    use hdf5::types::FixedAscii;
    use hdf5::File;
    use ndarray::Array1;
    use std::env::temp_dir;

    /// Fixture event file used elsewhere in the crate's tests.
    const TEST_FILE: &str = "./tests/test_data/HIFI00195790.nxs";
    /// Must match the real detector count in the fixture file.
    const N_SPEC: usize = 64;

    fn calculated_data() -> BatchData {
        let mut data = BatchData::new(TEST_FILE.to_string(), N_SPEC, 1, 1048576).unwrap();
        data.calculate().unwrap();
        data
    }

    /// A sample ValueLog<f32> for testing save_to_group (H5Type-compatible, no narrowing needed).
    fn sample_f32_log() -> ValueLog<f32> {
        ValueLog::<f32> {
            name: "temp".to_string(),
            time: Array1::from_vec(vec![0., 0.1, 0.2, 0.3]),
            value: Array1::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]),
            unit: "gas mark".to_string(),
        }
    }

    /// A sample ValueLog<f64> for testing save_with_narrowing (needs conversion to f32).
    fn sample_f64_log() -> ValueLog<f64> {
        ValueLog::<f64> {
            name: "pressure".to_string(),
            time: Array1::from_vec(vec![0., 0.1, 0.2, 0.3]),
            value: Array1::from_vec(vec![1.5f64, 2.5, 3.5, 4.5]),
            unit: "fathoms".to_string(),
        }
    }

    /// `save_to_group` should create a group named after the log, containing
    /// a `value_log` subgroup with `time` and `value` datasets.
    #[test]
    fn test_save_to_group_creates_expected_structure() {
        let log = sample_f32_log();

        let mut tmp_path = temp_dir();
        tmp_path.push("sample_log_save_to_group.nxs");
        let tmp = File::create(tmp_path).unwrap();
        let root = tmp.create_group("selog").unwrap();

        log.save_to_group(&root).unwrap();

        let log_group = root.group("temp").unwrap();
        let value_log = log_group.group("value_log").unwrap();

        let time: Array1<f32> = value_log.dataset("time").unwrap().read_1d().unwrap();
        let value: Array1<f32> = value_log.dataset("value").unwrap().read_1d().unwrap();

        // time is narrowed to f32 regardless of the source width
        let expected_time: Array1<f32> = log.time.map(|t| *t as f32);
        assert_eq!(time, expected_time);
        assert_eq!(value, log.value);

        let time_unit: FixedAscii<7> = value_log
            .dataset("time")
            .unwrap()
            .attr("units")
            .unwrap()
            .read_scalar()
            .unwrap();
        let unit: VarLenUnicode = value_log
            .dataset("value")
            .unwrap()
            .attr("units")
            .unwrap()
            .read_scalar()
            .unwrap();
        assert_eq!(time_unit.to_string(), "seconds".to_string());
        assert_eq!(unit.to_string(), "gas mark".to_string())
    }

    /// `save_with_narrowing` should narrow both the time array to f32 and the
    /// value array to the type's `NarrowTo32::Output` (e.g. f64 -> f32).
    #[test]
    fn test_save_with_narrowing_converts_values() {
        let log = sample_f64_log();
        let mut tmp_path = temp_dir();
        tmp_path.push("sample_log_save_narrowing.nxs");
        let tmp = File::create(tmp_path).unwrap();
        let root = tmp.create_group("selog").unwrap();

        log.save_with_narrowing(&root).unwrap();

        let log_group = root.group("pressure").unwrap();
        let value_log = log_group.group("value_log").unwrap();

        let time: Array1<f32> = value_log.dataset("time").unwrap().read_1d().unwrap();
        let value: Array1<f32> = value_log.dataset("value").unwrap().read_1d().unwrap();

        let expected_time: Array1<f32> = log.time.map(|t| *t as f32);
        let expected_value: Array1<f32> = log.value.map(|v| *v as f32);

        assert_eq!(time, expected_time);
        assert_eq!(value, expected_value);

        let time_unit: FixedAscii<7> = value_log
            .dataset("time")
            .unwrap()
            .attr("units")
            .unwrap()
            .read_scalar()
            .unwrap();
        let unit: VarLenUnicode = value_log
            .dataset("value")
            .unwrap()
            .attr("units")
            .unwrap()
            .read_scalar()
            .unwrap();
        assert_eq!(time_unit.to_string(), "seconds".to_string());
        assert_eq!(unit.to_string(), "fathoms".to_string())
    }

    /// `get_all_sample_logs` should return one `SampleLog` per name present
    /// in the dataset's `sample_log_names`, with no filters applied (since
    /// the calculated data has no time/log filters set by default).
    #[test]
    fn test_get_all_sample_logs_returns_all_logs_unfiltered() {
        let data = calculated_data();
        let logs = get_all_sample_logs(&data.dataset, &data.filters[0]).unwrap();

        assert_eq!(logs.len(), data.dataset.sample_log_names.len());
    }

    /// When filters are present, `get_all_sample_logs` should apply them to
    /// every returned log (i.e. the log's filtered result should differ from
    /// or match `apply_filters` called directly, depending on log content).
    #[test]
    fn test_get_all_sample_logs_applies_time_filters() {
        let mut data = calculated_data();
        let unfiltered_logs = get_all_sample_logs(&data.dataset, &data.filters[0]).unwrap();
        // Add a time filter matching the pattern used in other tests in the repo.
        data.add_time_filter(FilterIndex::Index(0), "test_filter".to_string(), 0.0, 1.0)
            .unwrap();
        data.calculate().unwrap();

        let logs = get_all_sample_logs(&data.dataset, &data.filters[0]);
        assert!(logs.is_ok());
        // sanity check: filtering shouldn't change the number of logs, only their contents
        let filtered_logs = logs.unwrap();
        assert_eq!(filtered_logs.len(), unfiltered_logs.len());
        for i in 0..filtered_logs.len() {
            assert!(filtered_logs[i] != unfiltered_logs[i])
        }
    }
}
