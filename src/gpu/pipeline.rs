use anyhow::{Context, Result};
use ndarray::Array3;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::OnceLock;
use wgpu::util::DeviceExt;

static GPU_CONTEXT: OnceLock<Option<GpuContext>> = OnceLock::new();

/// Uniform parameters passed to the WGSL histogram compute shader.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShaderParams {
    /// Minimum time threshold in microseconds.
    pub min_time: u32,
    /// Maximum time threshold in microseconds.
    pub max_time: u32,
    /// Number of bins in the time histogram.
    pub n_bins: u32,
    /// Number of detector spectra.
    pub n_spec: u32,
    /// Reciprocal of bin width (n_bins / (max_time - min_time)).
    pub inv_width: f32,
    /// Number of events in the current chunk.
    pub n_events: u32,
    /// Baseline minimum amplitude threshold.
    pub baseline_min_amp: f32,
    /// Flags bitmask: bit 0 = uniform min_amp, bit 1 = uniform period 0.
    pub flags: u32,
}

/// Global GPU execution context containing the device, command queue, and compiled compute pipeline.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub device_name: String,
    pub backend_name: String,
    pub device_type: String,
    pub is_destroyed: AtomicBool,
    pub cached_histogrammer: std::sync::Mutex<Option<GpuHistogrammer>>,
}

impl GpuContext {
    /// Retrieve a reference to the global GpuContext singleton, initializing it on first call.
    ///
    /// Returns
    /// -------
    /// Option<&'static GpuContext>
    ///     Reference to the initialized GpuContext, or None if no compatible GPU is available
    ///     or if the GPU context has been shut down.
    pub fn get() -> Option<&'static GpuContext> {
        let ctx = GPU_CONTEXT.get_or_init(|| Self::init().ok()).as_ref()?;
        if ctx.is_destroyed.load(Ordering::SeqCst) {
            None
        } else {
            Some(ctx)
        }
    }

    /// Explicitly destroy the active GPU device and release all GPU resources.
    /// This drains any pending work and destroys the underlying `wgpu::Device`,
    /// preventing driver shutdown crashes and unhandled C++ exceptions at process exit.
    pub fn shutdown() {
        if let Some(Some(ctx)) = GPU_CONTEXT.get() {
            if !ctx.is_destroyed.swap(true, Ordering::SeqCst) {
                if let Ok(mut lock) = ctx.cached_histogrammer.lock() {
                    *lock = None;
                }
                let _ = ctx.device.poll(wgpu::Maintain::Wait);
                ctx.device.destroy();
            }
        }
    }

    /// Acquire a cached GpuHistogrammer instance, or initialize a new one if dimensions have changed.
    pub fn acquire_histogrammer(
        &'static self,
        n_periods: usize,
        n_spec: usize,
        n_bins: usize,
        min_amps: &[f32],
        max_chunk_size: usize,
    ) -> Result<std::sync::MutexGuard<'static, Option<GpuHistogrammer>>> {
        let mut lock = self
            .cached_histogrammer
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to lock GPU buffer cache: {}", e))?;

        let reuse = if let Some(ref h) = *lock {
            h.matches(n_periods, n_spec, n_bins, max_chunk_size)
        } else {
            false
        };

        if reuse {
            if let Some(ref h) = *lock {
                h.reset(min_amps);
            }
        } else {
            let new_hist =
                GpuHistogrammer::new(self, n_periods, n_spec, n_bins, min_amps, max_chunk_size)?;
            *lock = Some(new_hist);
        }

        Ok(lock)
    }

    /// Initialize the GPU context by discovering an adapter, requesting a device and queue,
    /// and compiling the WGSL histogram shader.
    ///
    /// Returns
    /// -------
    /// Result<Self>
    ///     The initialized GpuContext instance, or an error if initialization failed.
    fn init() -> Result<Self> {
        let descriptor = wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        };
        let instance = wgpu::Instance::new(&descriptor.with_env());

        // If a discrete GPU (e.g. NVIDIA RTX / AMD Radeon) is present, prioritize it over integrated GPUs
        let mut discrete_adapter = None;
        for a in instance.enumerate_adapters(wgpu::Backends::PRIMARY) {
            if a.get_info().device_type == wgpu::DeviceType::DiscreteGpu {
                discrete_adapter = Some(a);
                break;
            }
        }

        let adapter = if let Some(a) = discrete_adapter {
            a
        } else {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            }))
            .context("No suitable GPU adapter found")?
        };

        let info = adapter.get_info();
        let device_name = info.name.clone();
        let backend_name = format!("{:?}", info.backend);
        let device_type = format!("{:?}", info.device_type);

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("MNeuEventLib GPU Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .context("Failed to request GPU device")?;

        let shader_source = include_str!("shader.wgsl");
        let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MNeuEventLib Histogram Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MNeuEventLib Bind Group Layout"),
            entries: &[
                // 0: params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: times storage
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2: specs storage
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3: amps storage
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 4: periods storage
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 5: min_amps storage
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 6: hist storage read_write
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MNeuEventLib Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MNeuEventLib Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &cs_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(GpuContext {
            device,
            queue,
            pipeline,
            bind_group_layout,
            device_name,
            backend_name,
            device_type,
            is_destroyed: AtomicBool::new(false),
            cached_histogrammer: std::sync::Mutex::new(None),
        })
    }
}

/// A double-buffered input slot for overlapping PCIe data transfers and GPU compute passes.
struct GpuChunkSlot {
    params_buffer: wgpu::Buffer,
    times_buffer: wgpu::Buffer,
    specs_buffer: wgpu::Buffer,
    amps_buffer: wgpu::Buffer,
    periods_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl GpuChunkSlot {
    fn new(
        ctx: &GpuContext,
        chunk_capacity: usize,
        min_amps_buffer: &wgpu::Buffer,
        hist_buffer: &wgpu::Buffer,
        label: &str,
    ) -> Self {
        let params_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("Params Buffer ({})", label)),
            size: std::mem::size_of::<ShaderParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let times_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("Times Buffer ({})", label)),
            size: (chunk_capacity * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let specs_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("Specs Buffer ({})", label)),
            size: (chunk_capacity * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let amps_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("Amps Buffer ({})", label)),
            size: (chunk_capacity * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let periods_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("Periods Buffer ({})", label)),
            size: (chunk_capacity * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Histogram Bind Group ({})", label)),
            layout: &ctx.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: times_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: specs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: amps_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: periods_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: min_amps_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: hist_buffer.as_entire_binding(),
                },
            ],
        });

        GpuChunkSlot {
            params_buffer,
            times_buffer,
            specs_buffer,
            amps_buffer,
            periods_buffer,
            bind_group,
        }
    }
}

/// GPU histogram accumulator managing double-buffered chunk slots, bind groups, and shader dispatch.
pub struct GpuHistogrammer {
    ctx: &'static GpuContext,
    slots: [GpuChunkSlot; 2],
    current_slot: AtomicUsize,
    min_amps_buffer: wgpu::Buffer,
    hist_buffer: wgpu::Buffer,
    staging_hist_buffer: wgpu::Buffer,
    hist_byte_size: u64,
    allocated_chunk_size: usize,
    n_periods: usize,
    n_spec: usize,
    n_bins: usize,
}

impl GpuHistogrammer {
    /// Allocate GPU buffers and bind groups for the given histogram dimensions and max chunk size.
    ///
    /// Parameters
    /// ----------
    /// ctx: &'static GpuContext
    ///     The active GPU context containing device and queue.
    /// n_periods: usize
    ///     The number of time periods in the experiment.
    /// n_spec: usize
    ///     The number of detector spectra.
    /// n_bins: usize
    ///     The number of bins in each histogram.
    /// min_amps: &[f32]
    ///     Per-spectrum pulse height discrimination thresholds.
    /// max_chunk_size: usize
    ///     The maximum chunk size (in events) that will be dispatched to the GPU.
    ///
    /// Returns
    /// -------
    /// Result<Self>
    ///     The allocated GpuHistogrammer instance ready for chunk dispatch.
    pub fn new(
        ctx: &'static GpuContext,
        n_periods: usize,
        n_spec: usize,
        n_bins: usize,
        min_amps: &[f32],
        max_chunk_size: usize,
    ) -> Result<Self> {
        let chunk_capacity = max_chunk_size.max(1);
        let hist_len = n_periods * n_spec * n_bins;
        let hist_byte_size = (hist_len * std::mem::size_of::<u32>()) as u64;

        let min_amps_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Min Amps Buffer"),
                contents: bytemuck::cast_slice(min_amps),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Initialize hist buffer with zeros
        let zero_hist = vec![0u8; hist_byte_size as usize];
        let hist_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Hist Buffer"),
                contents: &zero_hist,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        let staging_hist_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Hist Buffer"),
            size: hist_byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let slot0 =
            GpuChunkSlot::new(ctx, chunk_capacity, &min_amps_buffer, &hist_buffer, "Slot 0");
        let slot1 =
            GpuChunkSlot::new(ctx, chunk_capacity, &min_amps_buffer, &hist_buffer, "Slot 1");

        Ok(GpuHistogrammer {
            ctx,
            slots: [slot0, slot1],
            current_slot: AtomicUsize::new(0),
            min_amps_buffer,
            hist_buffer,
            staging_hist_buffer,
            hist_byte_size,
            allocated_chunk_size: chunk_capacity,
            n_periods,
            n_spec,
            n_bins,
        })
    }

    /// Check if the cached GPU buffers match the requested histogram shape and capacity.
    pub fn matches(
        &self,
        n_periods: usize,
        n_spec: usize,
        n_bins: usize,
        max_chunk_size: usize,
    ) -> bool {
        self.n_periods == n_periods
            && self.n_spec == n_spec
            && self.n_bins == n_bins
            && self.allocated_chunk_size >= max_chunk_size
    }

    /// Reset histogram buffer to zero and update discrimination thresholds without reallocation.
    pub fn reset(&self, min_amps: &[f32]) {
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Histogram Reset Encoder"),
            });
        encoder.clear_buffer(&self.hist_buffer, 0, None);
        self.ctx.queue.submit(Some(encoder.finish()));

        self.ctx
            .queue
            .write_buffer(&self.min_amps_buffer, 0, bytemuck::cast_slice(min_amps));
    }

    /// Upload event chunk data to GPU buffers and dispatch the compute shader workgroups asynchronously.
    ///
    /// Parameters
    /// ----------
    /// min_time: u32
    ///     Minimum time-of-flight threshold in microseconds.
    /// max_time: u32
    ///     Maximum time-of-flight threshold in microseconds.
    /// inv_width: f32
    ///     Reciprocal of the time bin width.
    /// times: &[u32]
    ///     Time-of-flight values for events in this chunk.
    /// specs: &[u32]
    ///     Spectrum/detector IDs for events in this chunk.
    /// amps: &[f32]
    ///     Pulse height amplitudes for events in this chunk.
    /// periods: &[u32]
    ///     Period numbers for events in this chunk.
    /// baseline_min_amp: f32
    ///     Baseline minimum amplitude value.
    /// has_uniform_min_amp: bool
    ///     True if all detectors share the identical baseline minimum amplitude.
    /// has_uniform_period: bool
    ///     True if all events in the dataset share period 0 with all weights valid.
    ///
    /// Returns
    /// -------
    /// Result<()>
    ///     Ok on successful dispatch, or an error if data exceeds buffer capacity.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_chunk(
        &self,
        min_time: u32,
        max_time: u32,
        inv_width: f32,
        times: &[u32],
        specs: &[u32],
        amps: &[f32],
        periods: &[u32],
        baseline_min_amp: f32,
        has_uniform_min_amp: bool,
        has_uniform_period: bool,
    ) -> Result<()> {
        let n_events = times.len();
        if n_events == 0 {
            return Ok(());
        }

        if n_events > self.allocated_chunk_size {
            anyhow::bail!(
                "Chunk size {} exceeds allocated GPU buffer capacity {}",
                n_events,
                self.allocated_chunk_size
            );
        }

        let slot_idx = self.current_slot.fetch_add(1, Ordering::Relaxed) % 2;
        let slot = &self.slots[slot_idx];

        let mut flags = 0u32;
        if has_uniform_min_amp {
            flags |= 1;
        }
        if has_uniform_period {
            flags |= 2;
        }

        let params = ShaderParams {
            min_time,
            max_time,
            n_bins: self.n_bins as u32,
            n_spec: self.n_spec as u32,
            inv_width,
            n_events: n_events as u32,
            baseline_min_amp,
            flags,
        };

        self.ctx
            .queue
            .write_buffer(&slot.params_buffer, 0, bytemuck::bytes_of(&params));
        self.ctx
            .queue
            .write_buffer(&slot.times_buffer, 0, bytemuck::cast_slice(times));
        self.ctx
            .queue
            .write_buffer(&slot.specs_buffer, 0, bytemuck::cast_slice(specs));
        self.ctx
            .queue
            .write_buffer(&slot.amps_buffer, 0, bytemuck::cast_slice(amps));

        if !has_uniform_period {
            self.ctx
                .queue
                .write_buffer(&slot.periods_buffer, 0, bytemuck::cast_slice(periods));
        }

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Compute Command Encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Histogram Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.ctx.pipeline);
            cpass.set_bind_group(0, &slot.bind_group, &[]);
            let workgroups = (n_events as u32).div_ceil(256);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.ctx.queue.submit(Some(encoder.finish()));
        if slot_idx == 1 {
            self.ctx.device.poll(wgpu::Maintain::Wait);
        }
        Ok(())
    }

    /// Copy the GPU histogram buffer to CPU staging memory and return the result.
    ///
    /// Parameters
    /// ----------
    /// n_periods: usize
    ///     The number of periods to reshape the output histogram into.
    ///
    /// Returns
    /// -------
    /// Result<(Array3<i32>, usize)>
    ///     A tuple containing the 3D histogram array and the total number of valid binned events.
    pub fn readback(&self, n_periods: usize) -> Result<(Array3<i32>, usize)> {
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Histogram Readback Encoder"),
            });

        encoder.copy_buffer_to_buffer(
            &self.hist_buffer,
            0,
            &self.staging_hist_buffer,
            0,
            self.hist_byte_size,
        );

        self.ctx.queue.submit(Some(encoder.finish()));

        let hist_slice = self.staging_hist_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();

        hist_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });

        self.ctx.device.poll(wgpu::Maintain::Wait);

        receiver
            .recv()?
            .map_err(|e| anyhow::anyhow!("Failed to map staging hist buffer: {:?}", e))?;

        let hist_data = hist_slice.get_mapped_range();
        let u32_slice: &[u32] = bytemuck::cast_slice(&hist_data);

        let mut total_count = 0usize;
        let mut i32_vec = Vec::with_capacity(u32_slice.len());
        for &val in u32_slice {
            total_count += val as usize;
            i32_vec.push(val as i32);
        }
        let hist = Array3::from_shape_vec((n_periods, self.n_spec, self.n_bins), i32_vec)?;

        drop(hist_data);
        self.staging_hist_buffer.unmap();

        Ok((hist, total_count))
    }
}
