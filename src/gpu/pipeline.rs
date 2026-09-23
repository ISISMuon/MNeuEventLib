use std::sync::OnceLock;
use anyhow::{Context, Result};
use ndarray::Array3;
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
    /// Alignment padding.
    pub _pad0: u32,
    /// Alignment padding.
    pub _pad1: u32,
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
}

impl GpuContext {
    /// Retrieve a reference to the global GpuContext singleton, initializing it on first call.
    ///
    /// Returns
    /// -------
    /// Option<&'static GpuContext>
    ///     Reference to the initialized GpuContext, or None if no compatible GPU is available.
    pub fn get() -> Option<&'static GpuContext> {
        GPU_CONTEXT
            .get_or_init(|| Self::init().ok())
            .as_ref()
    }

    /// Initialize the GPU context by discovering an adapter, requesting a device and queue,
    /// and compiling the WGSL histogram shader.
    ///
    /// Returns
    /// -------
    /// Result<Self>
    ///     The initialized GpuContext instance, or an error if initialization failed.
    fn init() -> Result<Self> {
        let instance = wgpu::Instance::default();

        // If a discrete GPU (e.g. NVIDIA RTX) is present, prioritize it over integrated GPUs
        let mut discrete_adapter = None;
        for a in instance.enumerate_adapters(wgpu::Backends::all()) {
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
                // 7: total_count storage read_write
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
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
        })
    }
}

/// GPU histogram accumulator managing buffers, bind groups, and shader dispatch.
pub struct GpuHistogrammer {
    ctx: &'static GpuContext,
    params_buffer: wgpu::Buffer,
    times_buffer: wgpu::Buffer,
    specs_buffer: wgpu::Buffer,
    amps_buffer: wgpu::Buffer,
    periods_buffer: wgpu::Buffer,
    _min_amps_buffer: wgpu::Buffer,
    hist_buffer: wgpu::Buffer,
    count_buffer: wgpu::Buffer,
    staging_hist_buffer: wgpu::Buffer,
    staging_count_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    hist_byte_size: u64,
    allocated_chunk_size: usize,
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

        let params_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Params Buffer"),
            size: std::mem::size_of::<ShaderParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let times_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Times Buffer"),
            size: (chunk_capacity * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let specs_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Specs Buffer"),
            size: (chunk_capacity * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let amps_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Amps Buffer"),
            size: (chunk_capacity * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let periods_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Periods Buffer"),
            size: (chunk_capacity * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let min_amps_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Min Amps Buffer"),
            contents: bytemuck::cast_slice(min_amps),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Initialize hist and count buffers with 0
        let zero_hist = vec![0u8; hist_byte_size as usize];
        let hist_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Hist Buffer"),
            contents: &zero_hist,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let zero_count = [0u8; 4];
        let count_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Count Buffer"),
            contents: &zero_count,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let staging_hist_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Hist Buffer"),
            size: hist_byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_count_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Count Buffer"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Histogram Bind Group"),
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
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: count_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(GpuHistogrammer {
            ctx,
            params_buffer,
            times_buffer,
            specs_buffer,
            amps_buffer,
            periods_buffer,
            _min_amps_buffer: min_amps_buffer,
            hist_buffer,
            count_buffer,
            staging_hist_buffer,
            staging_count_buffer,
            bind_group,
            hist_byte_size,
            allocated_chunk_size: chunk_capacity,
            n_spec,
            n_bins,
        })
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
    ///
    /// Returns
    /// -------
    /// Result<()>
    ///     Ok on successful dispatch, or an error if data exceeds buffer capacity.
    pub fn dispatch_chunk(
        &self,
        min_time: u32,
        max_time: u32,
        inv_width: f32,
        times: &[u32],
        specs: &[u32],
        amps: &[f32],
        periods: &[u32],
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

        let params = ShaderParams {
            min_time,
            max_time,
            n_bins: self.n_bins as u32,
            n_spec: self.n_spec as u32,
            inv_width,
            n_events: n_events as u32,
            _pad0: 0,
            _pad1: 0,
        };

        self.ctx
            .queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
        self.ctx
            .queue
            .write_buffer(&self.times_buffer, 0, bytemuck::cast_slice(times));
        self.ctx
            .queue
            .write_buffer(&self.specs_buffer, 0, bytemuck::cast_slice(specs));
        self.ctx
            .queue
            .write_buffer(&self.amps_buffer, 0, bytemuck::cast_slice(amps));
        self.ctx
            .queue
            .write_buffer(&self.periods_buffer, 0, bytemuck::cast_slice(periods));

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
            cpass.set_bind_group(0, &self.bind_group, &[]);
            let workgroups = ((n_events as u32) + 255) / 256;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.ctx.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// Copy the GPU histogram and count buffers to CPU staging memory and return the results.
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
    pub fn readback(self, n_periods: usize) -> Result<(Array3<i32>, usize)> {
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
        encoder.copy_buffer_to_buffer(
            &self.count_buffer,
            0,
            &self.staging_count_buffer,
            0,
            4,
        );

        self.ctx.queue.submit(Some(encoder.finish()));

        let hist_slice = self.staging_hist_buffer.slice(..);
        let count_slice = self.staging_count_buffer.slice(..);

        let (sender, receiver) = std::sync::mpsc::channel();
        let sender_count = sender.clone();

        hist_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });
        count_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender_count.send(res);
        });

        self.ctx.device.poll(wgpu::Maintain::Wait);

        receiver
            .recv()?
            .map_err(|e| anyhow::anyhow!("Failed to map staging hist buffer: {:?}", e))?;
        receiver
            .recv()?
            .map_err(|e| anyhow::anyhow!("Failed to map staging count buffer: {:?}", e))?;

        let hist_data = hist_slice.get_mapped_range();
        let count_data = count_slice.get_mapped_range();

        let u32_slice: &[u32] = bytemuck::cast_slice(&hist_data);
        let i32_vec: Vec<i32> = u32_slice.iter().map(|&x| x as i32).collect();
        let hist = Array3::from_shape_vec((n_periods, self.n_spec, self.n_bins), i32_vec)?;

        let total_count: u32 = *bytemuck::from_bytes(&count_data);

        drop(hist_data);
        drop(count_data);
        self.staging_hist_buffer.unmap();
        self.staging_count_buffer.unmap();

        Ok((hist, total_count as usize))
    }
}
