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

/// A struct containing the full file data for a save file.
#[allow(dead_code)]
pub struct WiMDAFile {
    /// The version of the histogram data format used.
    IDF_version: CopyData,
    /// The length of time spent counting good frames.
    good_duration: f32,
    /// The type of source.
    definition: CopyData,
    /// The number of frames which have been filtered out or vetoed.
    discarded_raw_frames: u32,
    /// The number of frames which have filtered out (but may or may not be vetoed)
    discarded_good_frames: u32,
    /// The amount of time between the first and last frame.
    duration: f32,
    /// The time of the last frame.
    end_time: usize, // stored here in ns from time zero
    experiment_identifier: CopyData,
    /// The number of frames which have not been filtered out or vetoed.
    good_frames: u32,
    /// The data collected by the instrument and related metadata.
    instrument: Instrument,
    /// The name of the instrument.
    name: CopyData,
    /// Any notes recorded during the experiment.
    notes: CopyData,
    /// Data about periods recorded during the experiment.
    periods: Periods,
    /// The number of frames which have not been filtered out.
    raw_frames: u32,
    /// The run number for the experiment.
    run_number: CopyData,
    /// Information about the sample used in the experiment.
    sample: CopyData,
    /// Information about sample logs recorded during the experiment.
    selog: Vec<SampleLog>,
    /// The time of the first frame.
    start_time: usize, // stored here in seconds from time zero
    /// The title of the experiment.
    title: CopyData,
    /// Information about the user performing the experiment.
    user_1: CopyData,
}

impl WiMDAFile {
    pub fn new(data: &Data) -> Result<WiMDAFile> {
        let unfiltered_frames = data.dataset.periods.size() as u32;
        // raw frames is all frames that have not been filtered,
        // good frames is all frames that have not been filtered or vetoed
        let raw_frames = data.results.n_frames.iter().sum::<u32>();
        let good_frames = data.results.n_good_frames.iter().sum::<u32>();
        let discarded_good_frames = unfiltered_frames - raw_frames;
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
            discarded_good_frames,
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
        let output = File::create(&filename)?;
        let hist_data = output.create_group("raw_data_1")?;
        let event_data = input_file.group("raw_data_1")?;
        add_nx_class(&hist_data, "NXentry")?;

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
        add_scalar(
            &hist_data,
            self.discarded_good_frames,
            "discarded_good_frames",
        )?;
        let duration = add_scalar(&hist_data, self.duration, "duration")?;
        add_str_attr::<7>(&duration, "seconds", "units")?;

        let start_time_string: VarLenUnicode =
            event_data.dataset("start_time")?.read()?.into_scalar();
        let unfiltered_start_time =
            DateTime::parse_from_str(start_time_string.as_ref(), "%Y-%m-%dT%H:%M:%S%z")?;

        // for some weird reason Chrono doesn't let you create a timedelta with more
        // than 1 second worth of nanoseconds, so we convert to seconds and nanoseconds
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

        // the raw_data_1/detector_1 group has different NX class, so we link the internal data
        let instrument_detector_1 = instrument_group.group("detector_1")?;
        let detector_1 = hist_data.create_group("detector_1")?;
        add_nx_class(&detector_1, "NXdata")?;

        for member in instrument_detector_1.member_names()? {
            hist_data.link_hard(
                &format!("instrument/detector_1/{member}"),
                &format!("detector_1/{member}"),
            )?;
        }

        match event_data.dataset("notes") {
            Ok(notes) => notes.copy_to(&hist_data, "notes")?,
            _ => {
                add_str_scalar::<1>(&hist_data, "", "notes")?;
            }
        };

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
        println!("Saved to output file {filename}");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interface::Data;
    use std::env::temp_dir;

    /// Fixture event file used elsewhere in the crate's tests.
    const TEST_FILE: &str = "./tests/test_data/HIFI00195790.nxs";

    fn calculated_data() -> Data {
        let mut data = Data::new(TEST_FILE.to_string(), 64, 1048576).unwrap();
        data.calculate().unwrap();
        data
    }

    /// `selog` should contain one `SampleLog` per name in the dataset's
    /// `sample_log_names`.
    #[test]
    fn test_wimda_file_new_selog_length() {
        let data = calculated_data();
        let wimda = WiMDAFile::new(&data).unwrap();

        assert_eq!(wimda.selog.len(), data.dataset.sample_log_names.len());
    }

    /// Saving a `WiMDAFile` should produce a valid file with the expected
    /// top-level `raw_data_1` group and key datasets.
    #[test]
    fn test_wimda_save_creates_expected_structure() {
        let _guard = crate::test_utils::lock_hdf5_test();
        let data = calculated_data();
        let wimda = WiMDAFile::new(&data).unwrap();

        let mut tmp_path = temp_dir();
        tmp_path.push("wimda_test_structure.nxs");
        let tmp_path_str = tmp_path.to_str().unwrap().to_string();

        wimda
            .save(tmp_path_str.clone(), &data.dataset.file)
            .unwrap();

        let output = File::open(tmp_path_str).unwrap();
        let hist_data = output.group("raw_data_1").unwrap();

        let nx_class: VarLenUnicode = hist_data.attr("NX_class").unwrap().read_scalar().unwrap();
        assert_eq!(nx_class.as_str(), "NXentry");

        let raw_frames: u32 = hist_data.dataset("raw_frames").unwrap().read_1d().unwrap()[0];
        assert_eq!(raw_frames, wimda.raw_frames);

        let good_frames: u32 = hist_data.dataset("good_frames").unwrap().read_1d().unwrap()[0];
        assert_eq!(good_frames, wimda.good_frames);

        let definition: hdf5::types::FixedAscii<8> =
            hist_data.dataset("definition").unwrap().read_1d().unwrap()[0];
        assert_eq!(definition.as_str(), "pulsedTD");

        // instrument and periods subgroups should exist and be linked correctly
        assert!(hist_data.group("instrument").is_ok());
        assert!(hist_data.group("periods").is_ok());
        assert!(hist_data.group("detector_1").is_ok());
        assert!(hist_data.group("selog").is_ok());
    }

    /// If the sample's `name` dataset is empty, saving should overwrite it
    /// with "unknown" (WiMDA crashes on an empty sample name).
    #[test]
    fn test_wimda_save_sample_name_defaults_to_unknown_when_empty() {
        let _guard = crate::test_utils::lock_hdf5_test();
        let data = calculated_data();
        let wimda = WiMDAFile::new(&data).unwrap();

        let mut tmp_path = temp_dir();
        tmp_path.push("wimda_test_sample_name.nxs");
        let tmp_path_str = tmp_path.to_str().unwrap().to_string();

        wimda
            .save(tmp_path_str.clone(), &data.dataset.file)
            .unwrap();

        let output = File::open(tmp_path_str).unwrap();
        let hist_data = output.group("raw_data_1").unwrap();
        let sample = hist_data.group("sample").unwrap();

        let name: VarLenUnicode = sample
            .dataset("name")
            .unwrap()
            .read()
            .unwrap()
            .into_scalar();
        assert!(!name.is_empty());
    }

    /// When the event file has no `user_1` group, saving should create a
    /// default one attributing the file to MNeuEventLib/RAL.
    #[test]
    fn test_wimda_save_user_1_default_when_missing() {
        let _guard = crate::test_utils::lock_hdf5_test();
        let data = calculated_data();
        let wimda = WiMDAFile::new(&data).unwrap();

        let mut tmp_path = temp_dir();
        tmp_path.push("wimda_test_user_1.nxs");
        let tmp_path_str = tmp_path.to_str().unwrap().to_string();

        wimda
            .save(tmp_path_str.clone(), &data.dataset.file)
            .unwrap();

        let output = File::open(tmp_path_str).unwrap();
        let hist_data = output.group("raw_data_1").unwrap();
        let user_1 = hist_data.group("user_1").unwrap();

        let name: hdf5::types::FixedAscii<12> =
            user_1.dataset("name").unwrap().read_1d().unwrap()[0];
        assert_eq!(name.as_str(), "MNeuEventLib");

        let affiliation: hdf5::types::FixedAscii<3> =
            user_1.dataset("affiliation").unwrap().read_1d().unwrap()[0];
        assert_eq!(affiliation.as_str(), "RAL");
    }
}
