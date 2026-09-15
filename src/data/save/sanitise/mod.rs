//! Code for saving outputs to files that
//! are compatible with Mantid.
//! This is done by filling in the missing metadata
//! in the output file based on a reference file.
//! This allows the file to be read by Mantid even if the
//! event file is incomplete.
pub mod nexus_data;
pub mod utils;
