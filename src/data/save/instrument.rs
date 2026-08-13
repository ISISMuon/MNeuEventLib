use anyhow::Result;
use hdf5::types::VarLenUnicode;
use hdf5::Group;
use ndarray::{Array1, Array3};

use crate::data::save::utils::*;
use crate::interface::Data;

#[allow(dead_code)]
pub struct Instrument {
    /// Information collected by the detectors.
    detector_1: Detector1,
    /// The name of the instrument.
    name: CopyData,
    /// Information about the instrument's source.
    source: CopyData,
}

impl Instrument {
    pub fn new(data: &Data) -> Instrument {
        Instrument {
            detector_1: Detector1::new(data),
            name: (),
            source: (),
        }
    }
}

impl Save for Instrument {
    fn save(&self, group: &Group, event_data: &Group) -> Result<()> {
        add_nx_class(group, "NXinstrument")?;

        let detector_1_group = group.create_group("detector_1")?;
        add_nx_class(&detector_1_group, "NXdetector")?;
        self.detector_1.save(&detector_1_group, event_data)?;

        let event_file_instrument = event_data.group("instrument")?;
        copy_scalar::<VarLenUnicode>(&event_file_instrument, group, "name")?;
        event_file_instrument
            .group("source")?
            .copy_to(group, "source")?;
        Ok(())
    }
}

/// A struct containing information about detected events.
struct Detector1 {
    /// Histogram data about counts recorded by the detectors.
    counts: CountsData,
    /// The minimum time for each bin of the histogram in microseconds.
    raw_time: Array1<f32>,
    /// The width of the bins in picoseconds.
    resolution: i32,
    /// The index for each detector.
    spectrum_index: Array1<i32>,
    /// The time which is used as 'zero' for the offset in all other times.
    time_zero: u32,
    /// The index of the periods for each period of the histogram.
    period_index: Array1<u32>,
}

/// Histogram data about counts recorded by the detectors.
struct CountsData {
    /// The histogram of counts, indexed [period index, spec index, bin]
    counts: Array3<i32>,
    /// The index of the first relevant bin.
    first_good_bin: u32,
    /// The index of the last relevant bin.
    last_good_bin: u32,
    /// The index of the bin corresponding to time_zero.
    t0_bin: u32,
}

impl Detector1 {
    fn new(data: &Data) -> Detector1 {
        let hist = &data.results;
        let width = (hist.max_time - hist.min_time) / hist.n_bins as f32;

        // these should be replaced when these attrs are added to event data
        let t0_bin: u32 = 0;
        let first_good_bin: u32 = 0;
        let last_good_bin = (hist.max_time / width).floor() as u32;

        let counts = CountsData {
            counts: hist.hist.clone(),
            first_good_bin,
            last_good_bin,
            t0_bin,
        };

        let raw_time = Array1::linspace(hist.min_time, hist.max_time, hist.n_bins + 1);

        let resolution = (width * 1e6) as i32;

        let n_spec = data.dataset.n_spec;
        let spectrum_index = Array1::from_vec((1..=n_spec as i32).collect());

        let n_periods: u32 = hist.hist.shape()[0] as u32;
        let period_index = Array1::from_iter(1..=n_periods);

        Detector1 {
            counts,
            raw_time,
            resolution,
            spectrum_index,
            time_zero: 0,
            period_index,
        }
    }
}

impl Save for Detector1 {
    fn save(&self, group: &Group, _: &Group) -> Result<()> {
        let counts = add_array(group, &self.counts.counts, "counts")?;
        add_str_attr::<44>(&counts, "period_index,spectrum_index,raw_time", "axes")?;
        add_attr(&counts, self.counts.t0_bin, "t0_bin")?;
        add_attr(&counts, self.counts.first_good_bin, "first_good_bin")?;
        add_attr(&counts, self.counts.last_good_bin, "last_good_bin")?;
        add_attr(&counts, 1, "signal")?;
        add_str_attr::<15>(&counts, "positron_counts", "long_name")?;

        let bins = add_array(group, &self.raw_time, "raw_time")?;
        add_str_attr::<4>(&bins, "time", "long_name")?;
        add_str_attr::<12>(&bins, "microseconds", "units")?;

        // min_time and max_time are in microseconds so we convert to picoseconds
        let res = add_scalar(group, self.resolution, "resolution")?;
        add_str_attr::<11>(&res, "picoseconds", "units")?;

        add_array(group, &self.spectrum_index, "spectrum_index")?;

        add_scalar(group, self.time_zero, "time_zero")?;

        add_array(group, &self.period_index, "period_index")?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interface::Data;

    /// Fixture event file already used by other test modules in the crate
    /// (see `data::nexus_data::tests`).
    const TEST_FILE: &str = "./tests/test_data/HIFI00195790.nxs";

    /// Build a `Data` object from the real test fixture and run `calculate()`
    /// so `results` (the `Histogram`) is populated before constructing an
    /// `Instrument` from it.
    ///
    /// NOTE: this assumes `Data::new` is `pub` (or `pub(crate)`) so it can be
    /// called from outside `interface.rs`. If it isn't, either make it pub
    /// or move these tests back into `interface.rs`'s own test module.
    fn calculated_data() -> Data {
        let mut data = Data::new(TEST_FILE.to_string(), 64, 1048576).unwrap();
        data.calculate().unwrap();
        data
    }

    /// `Instrument::new` should not panic, and its `detector_1` field should
    /// have a `spectrum_index` of length `n_spec`.
    #[test]
    fn test_instrument_new_spectrum_index_length() {
        let data = calculated_data();
        let instrument = Instrument::new(&data);

        assert_eq!(
            instrument.detector_1.spectrum_index.len(),
            data.dataset.n_spec
        );
    }

    /// `spectrum_index` should be 1-indexed: [1, 2, ..., n_spec].
    #[test]
    fn test_instrument_spectrum_index_values() {
        let data = calculated_data();
        let instrument = Instrument::new(&data);

        let expected: Array1<i32> = (1..=data.dataset.n_spec as i32).collect::<Vec<_>>().into();
        assert_eq!(instrument.detector_1.spectrum_index, expected);
    }

    /// `raw_time` should have length `n_bins + 1` and span [min_time, max_time].
    #[test]
    fn test_instrument_raw_time_bounds_and_length() {
        let data = calculated_data();
        let instrument = Instrument::new(&data);

        let raw_time = &instrument.detector_1.raw_time;
        assert_eq!(raw_time.len(), data.results.n_bins + 1);
        assert_eq!(*raw_time.first().unwrap(), data.results.min_time);
        assert_eq!(*raw_time.last().unwrap(), data.results.max_time);
    }

    /// `counts.counts` should be a clone of the histogram, matching shape
    /// and values exactly.
    #[test]
    fn test_instrument_counts_matches_histogram() {
        let data = calculated_data();
        let instrument = Instrument::new(&data);

        assert_eq!(instrument.detector_1.counts.counts, data.results.hist);
    }

    /// Explicitly recompute the expected bin values from a known histogram
    /// configuration and check them against `Detector1::new`.
    #[test]
    fn test_instrument_good_bins_explicit_values() {
        // min_time=0, max_time=10, n_bins=10 -> width = 1
        let mut data = calculated_data();
        data.results = crate::stats::Histogram::new(0., 10., 10);
        data.calculate().unwrap();

        let instrument = Instrument::new(&data);
        let counts = &instrument.detector_1.counts;

        // width = (10 - 0) / 10 = 1
        // t0_bin = floor(min_time / width) = 0
        // first_good_bin = ceil(min_time / width) = 0
        // last_good_bin = floor(max_time / width) = 10
        assert_eq!(counts.t0_bin, 0);
        assert_eq!(counts.first_good_bin, 0);
        assert_eq!(counts.last_good_bin, 10);
    }

    /// `resolution` should be the bin width in microseconds converted to
    /// picoseconds (width * 1e6), truncated to an integer.
    #[test]
    fn test_instrument_resolution_calculation() {
        let mut data = calculated_data();
        data.results = crate::stats::Histogram::new(0., 10., 10);
        data.calculate().unwrap();

        let instrument = Instrument::new(&data);

        let width = (10.0f32 - 0.0f32) / 10.0f32; // = 1.0
        let expected_resolution = (width * 1e6) as i32;
        assert_eq!(instrument.detector_1.resolution, expected_resolution);
    }

    /// Saving an `Instrument` should still succeed end-to-end (round trip
    /// through HDF5), confirming the public-field refactor doesn't break the
    /// `Save` implementation.
    #[test]
    fn test_instrument_save_round_trip() {
        use hdf5::File;
        use std::env::temp_dir;

        let data = calculated_data();
        let instrument = Instrument::new(&data);

        // todo: this needs to be done properly
        let mut tmp_path = temp_dir();
        tmp_path.push("instrument_test.nxs");
        let tmp = File::create(tmp_path).unwrap();
        let group = tmp.create_group("instrument").unwrap();
        let event_data = data.dataset.file.group("raw_data_1").unwrap();

        instrument.save(&group, &event_data).unwrap();

        let detector_1 = group.group("detector_1").unwrap();
        let counts = detector_1.dataset("counts").unwrap();
        assert_eq!(counts.shape(), data.results.hist.shape());
    }
}
