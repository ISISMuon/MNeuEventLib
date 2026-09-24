pub mod pipeline;

pub use pipeline::{GpuContext, GpuHistogrammer, ShaderParams};

use std::collections::HashMap;
use anyhow::{bail, Result};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Hardware execution target for histogram calculations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DevicePreference {
    /// Automatically select the fastest available device (GPU if available, else CPU).
    #[default]
    Auto,
    /// Multi-threaded CPU execution using Rayon.
    Cpu,
    /// GPU execution using wgpu compute shaders.
    Gpu,
    /// Concurrent hybrid execution splitting chunks across CPU and GPU.
    Hybrid,
}

impl DevicePreference {
    /// Parse a device preference string (case-insensitive, trimmed).
    ///
    /// Parameters
    /// ----------
    /// s: &str
    ///     The string representation of the target device ('auto', 'cpu', 'gpu', or 'hybrid').
    ///
    /// Returns
    /// -------
    /// Result<Self>
    ///     The parsed DevicePreference enum variant, or an error if unrecognized.
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().trim() {
            "auto" => Ok(DevicePreference::Auto),
            "cpu" => Ok(DevicePreference::Cpu),
            "gpu" => Ok(DevicePreference::Gpu),
            "hybrid" => Ok(DevicePreference::Hybrid),
            other => bail!("Unknown device '{other}'. Expected 'auto', 'cpu', 'gpu', or 'hybrid'."),
        }
    }

    /// Return the canonical string representation of the device preference.
    ///
    /// Returns
    /// -------
    /// &'static str
    ///     The string representation ('auto', 'cpu', 'gpu', or 'hybrid').
    pub fn as_str(&self) -> &'static str {
        match self {
            DevicePreference::Auto => "auto",
            DevicePreference::Cpu => "cpu",
            DevicePreference::Gpu => "gpu",
            DevicePreference::Hybrid => "hybrid",
        }
    }
}

/// Check whether a compatible GPU adapter is available on this system.
/// This queries the system for Vulkan, DirectX 12, or Metal adapters suitable
/// for running compute shaders.
///
/// Returns
/// -------
/// bool
///     True if a suitable Vulkan, DirectX 12, or Metal adapter can be initialized, False otherwise.
pub fn is_gpu_available() -> bool {
    GpuContext::get().is_some()
}

/// Retrieve device metadata for the active GPU adapter.
/// This inspects the initialized GPU adapter to report hardware specifications
/// including adapter name, driver backend, and device category.
///
/// Returns
/// -------
/// Option<HashMap<String, String>>
///     A map containing 'name', 'backend', and 'type' keys,
///     or None if no compatible GPU is available.
pub fn get_device_info() -> Option<HashMap<String, String>> {
    GpuContext::get().map(|ctx| {
        let mut map = HashMap::new();
        map.insert("name".to_string(), ctx.device_name.clone());
        map.insert("backend".to_string(), ctx.backend_name.clone());
        map.insert("type".to_string(), ctx.device_type.clone());
        map
    })
}

/// Check whether a compatible GPU adapter is available on this system.
/// This queries the graphics drivers and checks for Vulkan, DirectX 12, or Metal
/// hardware capable of executing WGSL compute shaders.
///
/// Returns
/// -------
/// bool
///     True if a compatible GPU (Vulkan, DirectX 12, or Metal) is available, False otherwise.
#[pyfunction(name = "is_gpu_available")]
pub fn is_gpu_available_py() -> bool {
    is_gpu_available()
}

/// Retrieve metadata about the active GPU device.
/// This inspects the initialized GPU adapter to report hardware specifications
/// including the adapter name, driver backend, and device category.
///
/// Returns
/// -------
/// dict or None
///     Dictionary containing 'name', 'backend', and 'type' of the active GPU adapter,
///     or None if no compatible GPU is available.
#[pyfunction(name = "get_device_info")]
pub fn get_device_info_py<'py>(py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
    if let Some(info) = get_device_info() {
        let dict = PyDict::new(py);
        for (k, v) in info {
            dict.set_item(k, v)?;
        }
        Ok(Some(dict))
    } else {
        Ok(None)
    }
}

/// Explicitly shutdown the active GPU device and release all GPU driver resources.
/// This destroys the wgpu::Device and drains GPU command queues before process exit,
/// preventing driver shutdown crashes and unhandled C++ exceptions.
pub fn cleanup_gpu() {
    pipeline::GpuContext::shutdown();
}

/// Explicitly shutdown the active GPU device and release all GPU driver resources.
/// This destroys the wgpu::Device and drains GPU command queues before process exit,
/// preventing driver shutdown crashes and unhandled C++ exceptions.
#[pyfunction(name = "cleanup_gpu")]
pub fn cleanup_gpu_py() {
    cleanup_gpu();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_preference_roundtrip() {
        assert_eq!(DevicePreference::from_str("auto").unwrap(), DevicePreference::Auto);
        assert_eq!(DevicePreference::from_str("CPU").unwrap(), DevicePreference::Cpu);
        assert_eq!(DevicePreference::from_str("  gpu  ").unwrap(), DevicePreference::Gpu);
        assert_eq!(DevicePreference::from_str("Hybrid").unwrap(), DevicePreference::Hybrid);

        assert_eq!(DevicePreference::Auto.as_str(), "auto");
        assert_eq!(DevicePreference::Cpu.as_str(), "cpu");
        assert_eq!(DevicePreference::Gpu.as_str(), "gpu");
        assert_eq!(DevicePreference::Hybrid.as_str(), "hybrid");
    }

    #[test]
    fn test_device_preference_invalid() {
        let err = DevicePreference::from_str("cuda").unwrap_err();
        assert!(err.to_string().contains("Unknown device 'cuda'"));

        let err2 = DevicePreference::from_str("").unwrap_err();
        assert!(err2.to_string().contains("Unknown device ''"));
    }

    #[test]
    fn test_is_gpu_available_and_device_info() {
        let available = is_gpu_available();
        let info = get_device_info();

        if available {
            assert!(info.is_some());
            let map = info.unwrap();
            assert!(map.contains_key("name"));
            assert!(map.contains_key("backend"));
            assert!(map.contains_key("type"));
            assert!(!map["name"].is_empty());
        } else {
            assert!(info.is_none());
        }
    }
}

