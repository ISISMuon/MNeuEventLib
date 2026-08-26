use core::str::FromStr;

use anyhow::Result;
use hdf5::types::VarLenUnicode;
use hdf5::Group;
use ndarray::{s, Array1};

use crate::data::save::utils::*;
use crate::Histogram;

pub struct Periods {
    /// The number of frames in each period.
    frames_requested: Array1<u32>,
    /// The names of each period.
    labels: String,
    /// The number of periods.
    number: u32,
    /// Output bit pattern on period card.
    output: Array1<i32>,
    /// The number of unfiltered frames in each period.
    raw_frames: Array1<u32>,
    /// The number of unfiltered and unvetoed frames in each period.
    good_frames: Array1<u32>,
    sequences: Array1<u32>,
    /// The number of events in each period.
    total_counts: Array1<f32>,
    /// The type of period.
    type_: Array1<i32>,
}

impl Periods {
    pub fn new(results: &Histogram) -> Periods {
        let n_periods = results.hist.shape()[0];
        let n_frames = Array1::from_vec(results.n_frames.clone());
        let n_good_frames = Array1::from_vec(results.n_good_frames.clone());

        let frames_requested = Array1::from_vec(results.n_frames.clone());
        let labels = (0..n_periods)
            .map(|n| format!("period {}", n + 1))
            .collect::<Vec<String>>()
            .join(",");
        // this seems to be unused or needs to be read from data (but doesn't exist in data)
        let output = Array1::from_vec(vec![0; n_periods]);
        let total_counts: Array1<f32> = (0..n_periods)
            .map(|n| {
                let slice = s![n, .., ..];
                results.hist.slice(slice).sum() as f32 / 1e6
            })
            .collect();
        let type_ = Array1::from_vec(vec![1; n_periods]);

        Periods {
            frames_requested,
            labels,
            number: n_periods as u32,
            output,
            raw_frames: n_frames.clone(),
            good_frames: n_good_frames,
            sequences: n_frames,
            total_counts,
            type_,
        }
    }
}

impl Save for Periods {
    fn save(&self, group: &Group, _: &Group) -> Result<()> {
        add_array(group, &self.frames_requested, "frames_requested")?;
        let labels = VarLenUnicode::from_str(&self.labels)?;
        add_scalar(group, labels, "labels")?;
        let label_dataset = group.dataset("labels")?;
        add_str_attr::<1>(&label_dataset, ",", "separator")?;

        add_scalar(group, self.number, "number")?;
        add_array(group, &self.output, "output")?;
        add_array(group, &self.raw_frames, "raw_frames")?;
        add_array(group, &self.good_frames, "good_frames")?;
        add_array(group, &self.sequences, "sequences")?;

        add_array(group, &self.total_counts, "total_counts")?;

        add_array(group, &self.type_, "type")?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interface::Data;

    /// Fixture event file used elsewhere in the crate's tests.
    const TEST_FILE: &str = "./tests/test_data/HIFI00195790.nxs";

    fn calculated_data() -> Histogram {
        let mut data = Data::new(TEST_FILE.to_string(), 64, 1048576).unwrap();
        data.calculate().unwrap();
        data.results
    }

    /// `Periods::new` should not panic, and `number` should match the
    /// number of periods present in the calculated histogram.
    #[test]
    fn test_periods_new_number_matches_histogram() {
        let data = calculated_data();
        let periods = Periods::new(&data);

        let n_periods = data.hist.shape()[0];
        assert_eq!(periods.number, n_periods as u32);
    }

    /// `frames_requested`, `raw_frames`, and `sequences` should all have a
    /// length equal to the number of periods.
    #[test]
    fn test_periods_arrays_length_matches_number() {
        let data = calculated_data();
        let periods = Periods::new(&data);

        assert_eq!(periods.frames_requested.len(), periods.number as usize);
        assert_eq!(periods.raw_frames.len(), periods.number as usize);
        assert_eq!(periods.sequences.len(), periods.number as usize);
        assert_eq!(periods.total_counts.len(), periods.number as usize);
        assert_eq!(periods.output.len(), periods.number as usize);
        assert_eq!(periods.type_.len(), periods.number as usize);
    }

    /// `labels` should be a comma-separated list of "period N" for
    /// N = 1..=n_periods.
    #[test]
    fn test_periods_labels_format() {
        let data = calculated_data();
        let periods = Periods::new(&data);

        let n_periods = data.hist.shape()[0];
        let expected = (0..n_periods)
            .map(|n| format!("period {}", n + 1))
            .collect::<Vec<String>>()
            .join(",");

        assert_eq!(periods.labels, expected);
    }

    /// Test the period labels look as expected.
    #[test]
    fn test_periods_labels() {
        let data = calculated_data();
        assert_eq!(
            data.hist.shape()[0],
            2,
            "fixture is expected to have exactly two periods"
        );

        let periods = Periods::new(&data);
        assert_eq!(periods.labels, "period 1,period 2");
    }

    /// `total_counts` for each period should equal the sum of that period's
    /// histogram slice, scaled down by 1e6 (as `Periods::new` does).
    #[test]
    fn test_periods_total_counts_matches_histogram_sum() {
        let data = calculated_data();
        let periods = Periods::new(&data);

        let n_periods = data.hist.shape()[0];
        for n in 0..n_periods {
            let slice = data.hist.slice(ndarray::s![n, .., ..]);
            let expected = slice.sum() as f32 / 1e6;
            assert_eq!(periods.total_counts[n], expected);
        }
    }

    /// Saving a `Periods` object should produce the expected datasets and
    /// attributes in the HDF5 group.
    #[test]
    fn test_periods_save_round_trip() {
        use hdf5::File;
        use std::env::temp_dir;

        let data = calculated_data();
        let periods = Periods::new(&data);

        let mut tmp_path = temp_dir();
        tmp_path.push("periods_test.nxs");
        let tmp = File::create(tmp_path).unwrap();
        let group = tmp.create_group("periods").unwrap();
        let event_data = File::open(TEST_FILE).unwrap().group("raw_data_1").unwrap();

        let result = periods.save(&group, &event_data);
        assert!(result.is_ok());

        // number
        let number: u32 = group.dataset("number").unwrap().read_1d().unwrap()[0];
        assert_eq!(number, periods.number);

        // labels + separator attribute
        let labels: &hdf5::types::VarLenUnicode =
            &group.dataset("labels").unwrap().read_1d().unwrap()[0];
        assert_eq!(labels.as_str(), periods.labels);

        let separator: hdf5::types::FixedAscii<1> = group
            .dataset("labels")
            .unwrap()
            .attr("separator")
            .unwrap()
            .read_scalar()
            .unwrap();
        assert_eq!(separator.as_str(), ",");

        // frames_requested / raw_frames / sequences arrays round-trip correctly
        let frames_requested: Array1<u32> = group
            .dataset("frames_requested")
            .unwrap()
            .read_1d()
            .unwrap();
        assert_eq!(frames_requested, periods.frames_requested);

        let raw_frames: Array1<u32> = group.dataset("raw_frames").unwrap().read_1d().unwrap();
        assert_eq!(raw_frames, periods.raw_frames);

        let sequences: Array1<u32> = group.dataset("sequences").unwrap().read_1d().unwrap();
        assert_eq!(sequences, periods.sequences);

        let total_counts: Array1<f32> = group.dataset("total_counts").unwrap().read_1d().unwrap();
        assert_eq!(total_counts, periods.total_counts);
    }
}
