use core::str::FromStr;

use anyhow::{Error, Result};
use hdf5::types::{FixedAscii, VarLenUnicode};
use hdf5::{File, Group, H5Type, Location};
use ndarray::{arr0, s, Array, Array1, Array3, Dimension};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::data::{SampleLog, ValueLog};
use crate::interface::Data;
use crate::utils::NarrowTo32;

// placeholder type for data that will just be copied into the nexus file
type CopyData<T> = T;

pub trait SaveFile {
    /// Save a Data object to file.
    fn save(data: &Data, filename: String) -> Result<()>;
}

trait Save {
    /// Save a Data object to a HDF5 group.
    fn save(data: &Data, group: &Group) -> Result<()>;
}

/// Create a dataset with a scalar in it.
fn add_scalar<T: H5Type>(group: &Group, scalar: T, name: &str) -> Result<()> {
    let data: Array1<T> = Array1::from_vec(vec![scalar]);
    add_array(group, &data, name)
}

/// Create a dataset with a scalar in it.
fn add_array<T: H5Type, D: Dimension>(
    group: &Group,
    array: &Array<T, D>,
    name: &str,
) -> Result<()> {
    let builder = group.new_dataset_builder();
    let builder = builder.with_data(array);
    builder.create(name)?;
    Ok(())
}

fn add_str_scalar<const LEN: usize>(group: &Group, scalar: &str, name: &str) -> Result<()> {
    let string = FixedAscii::<LEN>::from_ascii(scalar)?;
    add_scalar(group, string, name)
}

fn copy_scalar<T: H5Type>(from: &Group, to: &Group, item: &str) -> Result<()> {
    let data: T = from.dataset(item)?.read()?.into_scalar();
    add_scalar(to, data, item)
}

fn copy_time(from: &Group, to: &Group, item: &str) -> Result<()> {
    // the event files use RFC3339 formatted time,
    // but the histogram files don't have a timezone
    let data: VarLenUnicode = from.dataset(item)?.read()?.into_scalar();
    let date_string = data.as_str();

    let output = &date_string[..19];

    add_str_scalar::<20>(to, &output, item)
}

/// Set an attribute of a group.
fn add_attr<T: H5Type>(loc: &Location, data: T, name: &str) -> Result<()> {
    let builder = loc.new_attr_builder();
    let scalar = arr0(data);
    let builder = builder.with_data(&scalar);
    builder.create(name)?;
    Ok(())
}

fn add_str_attr<const LEN: usize>(loc: &Location, data: &str, name: &str) -> Result<()> {
    let string = FixedAscii::<LEN>::from_ascii(data)?;
    add_attr(loc, string, name)
}

/// Set the NX_class of a group.
fn add_nx_class(group: &Group, class: &str) -> Result<()> {
    let string = VarLenUnicode::from_str(class)?;
    add_attr(group, string, "NX_class")
}

pub struct WiMDAFile {
    file: File,
    IDF_version: CopyData<i32>,
    count_duration: f32,
    definition: CopyData<String>,
    discarded_raw_frames: i32,
    duration: f32,
    end_time: String,
    experiment_identifier: CopyData<String>,
    good_frames: i32,
    instrument: Instrument,
    name: CopyData<String>,
    notes: CopyData<String>,
    periods: Periods,
    raw_frames: i32,
    run_number: CopyData<i32>,
    sample: CopyData<Group>,
    selog: Vec<SampleLog>,
    start_time: String,
    title: CopyData<String>,
    user_1: CopyData<User1>,
}

impl SaveFile for WiMDAFile {
    fn save(data: &Data, filename: String) -> Result<()> {
        let event_data = data.dataset.file.group("raw_data_1")?;

        let file = File::create(filename)?;
        let hist_data = file.create_group("raw_data_1")?;
        add_nx_class(&hist_data, "NXentry")?;

        let unfiltered_frames = data.dataset.periods.size() as u32;
        // raw frames is all frames that have not been filtered,
        // good frames is all frames that have not been filtered or vetoed
        let raw_frames = data.results.n_frames.iter().sum::<u32>();
        let good_frames = data.results.n_good_frames.iter().sum::<u32>();
        //let discarded_good_frames = unfiltered_frames - raw_frames;
        let discarded_raw_frames = unfiltered_frames - good_frames;

        copy_scalar::<i32>(&event_data, &hist_data, "IDF_version")?;
        // todo: does count_duration need calculating?
        match event_data.dataset("count_duration") {
            Ok(duration) => duration.copy_to(&hist_data, "count_duration")?,
            Err(_) => {
                let count_duration = good_frames as f32 * 0.025; // at ISIS we expect 4 pulses over 100ms
                add_scalar(&hist_data, count_duration, "count_duration")?;
            }
        }
        let count_duration = hist_data.dataset("count_duration")?;
        add_str_attr::<7>(&count_duration, "seconds", "units")?;

        add_str_scalar::<8>(&hist_data, "pulsedTD", "definition")?;
        add_scalar(&hist_data, discarded_raw_frames, "discarded_raw_frames")?;
        // todo: duration to calculated end_time - start_time?
        hist_data
            .dataset("count_duration")?
            .copy_to(&hist_data, "duration")?;

        // todo: end_time to time of last good event?
        copy_time(&event_data, &hist_data, "end_time")?;
        copy_scalar::<VarLenUnicode>(&event_data, &hist_data, "experiment_identifier")?;
        add_scalar(&hist_data, good_frames, "good_frames")?;
        let instrument_group = hist_data.create_group("instrument")?;
        Instrument::save(data, &instrument_group)?;
        add_scalar(&hist_data, VarLenUnicode::from_str("name")?, "name")?;
        hist_data.link_hard("instrument/detector_1", "detector_1")?;

        match event_data.dataset("notes") {
            Ok(notes) => notes.copy_to(&hist_data, "notes")?,
            _ => add_scalar(&hist_data, FixedAscii::<1>::new(), "notes")?,
        };
        // todo: remove periods if empty?
        let period_group = hist_data.create_group("periods")?;
        add_nx_class(&period_group, "IXperiod")?;
        Periods::save(data, &period_group)?;

        add_scalar(&hist_data, raw_frames, "raw_frames")?;
        copy_scalar::<i32>(&event_data, &hist_data, "run_number")?;
        event_data.group("sample")?.copy_to(&hist_data, "sample")?;
        let sample = hist_data.group("sample")?;
        // WiMDA crashes if sample name is not provided
        let sample_name = sample.dataset("name")?;
        let sample_name_data: VarLenUnicode = sample_name.read()?.into_scalar();
        if sample_name_data.len() == 0 {
            sample
                .dataset("name")?
                .write(&arr0(VarLenUnicode::from_str("unknown")?))?;
        }
        add_scalar(&sample, 0, "height")?;
        add_scalar(&sample, FixedAscii::<1>::new(), "id")?;
        add_scalar(&sample, 0, "width")?;
        // todo: actually filter sample logs instead of just copying
        hist_data.create_group("selog")?;
        let selog = hist_data.group("selog")?;
        save_sample_logs(data, &selog)?;
        //event_data.group("selog")?.copy_to(&hist_data, "selog")?;
        // todo: start_time to time of first good event?
        copy_time(&event_data, &hist_data, "start_time")?;
        copy_scalar::<VarLenUnicode>(&event_data, &hist_data, "title")?;
        User1::save(data, &hist_data)?;

        Ok(())
    }
}

struct Instrument {
    detector_1: Detector1,
    name: CopyData<String>,
    source: CopyData<Group>,
}

impl Save for Instrument {
    fn save(data: &Data, group: &Group) -> Result<()> {
        add_nx_class(group, "NXinstrument")?;

        let detector_1_group = group.create_group("detector_1")?;
        add_nx_class(&detector_1_group, "NXdetector")?;
        Detector1::save(data, &detector_1_group)?;

        let event_file_instrument = data.dataset.file.group("raw_data_1")?.group("instrument")?;
        copy_scalar::<VarLenUnicode>(&event_file_instrument, &group, "name")?;
        event_file_instrument
            .group("source")?
            .copy_to(group, "source")?;
        Ok(())
    }
}

struct Detector1 {
    counts: Array3<i32>,
    raw_time: Array1<f32>,
    resolution: i32,
    spectrum_index: Array1<i32>,
}

impl Save for Detector1 {
    fn save(data: &Data, group: &Group) -> Result<()> {
        let hist = &data.results;
        let width = (hist.max_time - hist.min_time) / hist.n_bins as f32;

        add_array(group, &hist.hist, "counts")?;
        let counts = group.dataset("counts")?;
        add_str_attr::<44>(&counts, "period_index,spectrum_index,raw_time", "axes")?;
        add_attr(&counts, (hist.min_time / width).floor(), "t0_bin")?;
        add_attr(&counts, (hist.min_time / width).ceil(), "first_good_bin")?;
        add_attr(
            &counts,
            (hist.max_time / width).floor() - 1.,
            "last_good_bin",
        )?;
        add_str_attr::<6>(&counts, "counts", "long_name")?;

        let bins_builder = group.new_dataset_builder();
        let bins = Array1::linspace(hist.min_time, hist.max_time, hist.n_bins + 1);
        let bins_builder = bins_builder.with_data(&bins);
        bins_builder.create("raw_time")?;
        let bins = group.dataset("raw_time")?;
        add_str_attr::<4>(&bins, "time", "long_name")?;
        add_str_attr::<12>(&bins, "microseconds", "units")?;

        // min_time and max_time are in microseconds so we convert to picoseconds
        add_scalar(group, (width * 1e6) as i32, "resolution")?;
        let res = group.dataset("resolution")?;
        add_str_attr::<11>(&res, "picoseconds", "units")?;

        let n_spec = data.dataset.n_spec;
        let specs_builder = group.new_dataset_builder();
        let specs: Array1<i32> = Array1::from_vec((1..=n_spec as i32).collect());
        let specs_builder = specs_builder.with_data(&specs);
        specs_builder.create("spectrum_index")?;

        Ok(())
    }
}

struct Periods {
    frames_requested: Array1<i32>,
    labels: String,
    number: i32,
    output: Array1<i32>,
    raw_frames: Array1<i32>,
    sequences: Array1<i32>,
    total_counts: Array1<f32>,
    type_: Array1<i32>,
}

impl Save for Periods {
    fn save(data: &Data, group: &Group) -> Result<()> {
        let n_periods = data.results.hist.shape()[0];
        let n_frames = Array1::from_vec(data.results.n_frames.clone());

        add_array(group, &n_frames, "frames_requested")?;
        let labels = (0..n_periods)
            .map(|n| format!("period {}", n + 1))
            .collect::<Vec<String>>()
            .join(",");
        let labels = VarLenUnicode::from_str(&labels)?;
        add_scalar(group, labels, "labels")?;
        let label_dataset = group.dataset("labels")?;
        add_str_attr::<1>(&label_dataset, ",", "separator")?;

        add_scalar(group, n_periods, "number")?;
        add_array(group, &Array1::from_vec(vec![0; n_periods]), "output")?;
        add_array(group, &n_frames, "raw_frames")?;
        add_array(group, &n_frames, "sequences")?;

        let total_counts: Array1<f32> = (0..n_periods)
            .map(|n| {
                let slice = s![n, .., ..];
                data.results.hist.slice(slice).sum() as f32 / 1e6
            })
            .collect();
        add_array(group, &total_counts, "total_counts")?;

        let types = Array1::<i32>::from_vec(vec![1; n_periods]);
        let type_builder = group.new_dataset_builder();
        let type_builder = type_builder.with_data(&types);
        type_builder.create("type")?;

        Ok(())
    }
}

struct User1 {
    name: String,
    affiliation: String,
}

impl Save for User1 {
    fn save(data: &Data, group: &Group) -> Result<()> {
        let event_user1 = data.dataset.file.group("raw_data_1")?.group("user_1");
        match event_user1 {
            Ok(user) => user.copy_to(group, "user_1")?,
            Err(_) => {
                let user_1 = group.create_group("user_1")?;
                add_nx_class(&user_1, "NXuser")?;
                add_scalar(&user_1, VarLenUnicode::from_str("MNeuEventLib")?, "name")?;
                add_scalar(&user_1, VarLenUnicode::from_str("RAL")?, "affiliation")?;
            }
        };
        Ok(())
    }
}

/// Save the log to a HDF5 Group
impl SampleLog {
    pub fn save_to_group(&self, group: &Group) -> Result<()> {
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
        let times: Array1<f32> = self.time.map(|t| *t as f32);
        let values = self.value.map(|v| v.narrow());
        add_array(group, &times, "time")?;
        add_array(group, &values, "value")?;
        Ok(())
    }
}

impl<T> ValueLog<T>
where
    T: H5Type,
{
    /// Save a log to a group.
    pub fn save_to_group(&self, group: &Group) -> Result<()> {
        let times: Array1<f32> = self.time.map(|t| *t as f32);
        add_array(group, &times, "time")?;
        add_array(group, &self.value, "value")?;
        Ok(())
    }
}

fn save_sample_logs(data: &Data, group: &Group) -> Result<()> {
    let event_data = &data.dataset;
    let filters = &data.filters;
    let (mut time_starts, mut time_ends) = filters.get_time_filter_times();

    let log_names = filters.get_required_log_names();
    let value_logs = match event_data.get_sample_logs(log_names) {
        Ok(logs) => logs,
        Err(info) => return Err(Error::msg(format!("Failed to get logs: {info}"))),
    };
    let (log_starts, log_ends) = filters.get_log_filter_times(value_logs);

    time_starts.extend(log_starts);
    time_ends.extend(log_ends);

    event_data.sample_log_names.par_iter().for_each(|name| {
        let mut log = event_data.get_sample_log(name).unwrap();
        log = match time_starts.is_empty() {
            true => log,
            false => log.apply_filters(&time_starts, &time_ends),
        };
        group.create_group(&name).unwrap();
        let log_group = group.group(&name).unwrap();
        log.save_to_group(&log_group).unwrap();
    });
    Ok(())
}
