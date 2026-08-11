use std::str::FromStr;

use anyhow::Result;
use chrono::{DateTime, TimeDelta};
use hdf5::types::VarLenUnicode;
use hdf5::File;
use ndarray::arr0;

use crate::consts::S_TO_NS;
use crate::data::save::sample_logs::get_all_sample_logs;
use crate::data::save::utils::*;
use crate::data::save::{Instrument, Periods};
use crate::data::SampleLog;
use crate::interface::Data;

#[allow(dead_code)]
pub struct WiMDAFile {
    IDF_version: CopyData,
    good_duration: f32,
    definition: CopyData,
    discarded_raw_frames: u32,
    duration: f32,
    end_time: usize, // stored here in ns from time zero
    experiment_identifier: CopyData,
    good_frames: u32,
    instrument: Instrument,
    name: CopyData,
    notes: CopyData,
    periods: Periods,
    raw_frames: u32,
    run_number: CopyData,
    sample: CopyData,
    selog: Vec<SampleLog>,
    start_time: usize, // stored here in seconds from time zero
    title: CopyData,
    user_1: CopyData,
}

impl WiMDAFile {
    pub fn new(data: &Data) -> Result<WiMDAFile> {
        let unfiltered_frames = data.dataset.periods.size() as u32;
        // raw frames is all frames that have not been filtered,
        // good frames is all frames that have not been filtered or vetoed
        let raw_frames = data.results.n_frames.iter().sum::<u32>();
        let good_frames = data.results.n_good_frames.iter().sum::<u32>();
        //let discarded_good_frames = unfiltered_frames - raw_frames;
        let discarded_raw_frames = unfiltered_frames - good_frames;

        let good_duration = good_frames as f32 * 0.025;
        let end_time = data.results.end_time;
        let start_time = data.results.start_time;
        let duration = (end_time - start_time) as f32 / S_TO_NS as f32;

        let instrument = Instrument::new(data);

        let periods = Periods::new(data);

        let selog = get_all_sample_logs(data)?;

        Ok(WiMDAFile {
            IDF_version: (),
            good_duration,
            definition: (),
            discarded_raw_frames,
            duration,
            end_time,
            experiment_identifier: (),
            good_frames,
            instrument,
            name: (),
            notes: (),
            periods,
            raw_frames,
            run_number: (),
            sample: (),
            selog,
            start_time,
            title: (),
            user_1: (),
        })
    }
}

impl SaveFile for WiMDAFile {
    fn save(&self, filename: String, input_file: &File) -> Result<()> {
        let output = File::create(filename)?;
        let hist_data = output.create_group("raw_data_1")?;
        let event_data = input_file.group("raw_data_1")?;
        add_nx_class(&hist_data, "NXEntry")?;

        copy_scalar::<i32>(&event_data, &hist_data, "IDF_version")?;

        match event_data.dataset("good_duration") {
            Ok(duration) => duration.copy_to(&hist_data, "good_duration")?,
            Err(_) => {
                add_scalar(&hist_data, self.good_duration, "good_duration")?;
            }
        };
        let good_duration = hist_data.dataset("good_duration")?;
        add_str_attr::<7>(&good_duration, "seconds", "units")?;

        add_str_scalar::<8>(&hist_data, "pulsedTD", "definition")?;
        add_scalar(
            &hist_data,
            self.discarded_raw_frames,
            "discarded_raw_frames",
        )?;
        let duration = add_scalar(&hist_data, self.duration, "duration")?;
        add_str_attr::<7>(&duration, "seconds", "units")?;

        let start_time_string: VarLenUnicode =
            event_data.dataset("start_time")?.read()?.into_scalar();
        let unfiltered_start_time =
            DateTime::parse_from_str(&start_time_string.to_string(), "%Y-%m-%dT%H:%M:%S%z")?;

        let start_time = unfiltered_start_time
            + TimeDelta::new(
                (self.start_time / 1_000_000_000) as i64,
                (self.start_time % 1_000_000_000) as u32,
            )
            .unwrap();
        let end_time = unfiltered_start_time
            + TimeDelta::new(
                (self.end_time / 1_000_000_000) as i64,
                (self.end_time % 1_000_000_000) as u32,
            )
            .unwrap();

        add_str_scalar::<20>(
            &hist_data,
            &start_time.format("%Y-%m-%dT%H:%M:%S").to_string(),
            "start_time",
        )?;
        add_str_scalar::<20>(
            &hist_data,
            &end_time.format("%Y-%m-%dT%H:%M:%S").to_string(),
            "end_time",
        )?;

        copy_scalar::<VarLenUnicode>(&event_data, &hist_data, "experiment_identifier")?;

        add_scalar(&hist_data, self.good_frames, "good_frames")?;

        let instrument_group = hist_data.create_group("instrument")?;
        self.instrument.save(&instrument_group, &event_data)?;

        copy_scalar::<VarLenUnicode>(&instrument_group, &hist_data, "name")?;
        hist_data.link_hard("instrument/detector_1", "detector_1")?;

        match event_data.dataset("notes") {
            Ok(notes) => notes.copy_to(&hist_data, "notes")?,
            _ => {
                add_str_scalar::<1>(&hist_data, "", "notes")?;
            }
        };
        // todo: remove periods if empty?
        let period_group = hist_data.create_group("periods")?;
        add_nx_class(&period_group, "IXperiod")?;
        self.periods.save(&period_group, &event_data)?;

        add_scalar(&hist_data, self.raw_frames, "raw_frames")?;
        copy_scalar::<i32>(&event_data, &hist_data, "run_number")?;
        event_data.group("sample")?.copy_to(&hist_data, "sample")?;
        let sample = hist_data.group("sample")?;
        // WiMDA crashes if sample name is not provided
        let sample_name = sample.dataset("name")?;
        let sample_name_data: VarLenUnicode = sample_name.read()?.into_scalar();
        if sample_name_data.is_empty() {
            sample
                .dataset("name")?
                .write(&arr0(VarLenUnicode::from_str("unknown")?))?;
        }
        add_scalar(&sample, 0, "height")?;
        add_str_scalar::<1>(&sample, "", "id")?;
        add_scalar(&sample, 0, "width")?;

        let selog_group = hist_data.create_group("selog")?;
        for sample_log in &self.selog {
            sample_log.save(&selog_group, &event_data)?
        }

        copy_scalar::<VarLenUnicode>(&event_data, &hist_data, "title")?;

        let event_user1 = event_data.group("user_1");
        match event_user1 {
            Ok(user) => user.copy_to(&hist_data, "user_1")?,
            Err(_) => {
                let user_1 = hist_data.create_group("user_1")?;
                add_nx_class(&user_1, "NXuser")?;
                add_str_scalar::<12>(&user_1, "MNeuEventLib", "name")?;
                add_str_scalar::<3>(&user_1, "RAL", "affiliation")?;
            }
        };
        Ok(())
    }
}
