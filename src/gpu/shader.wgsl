struct Params {
    min_time: u32,
    max_time: u32,
    n_bins: u32,
    n_spec: u32,
    inv_width: f32,
    n_events: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0)
var<uniform> params: Params;

@group(0) @binding(1)
var<storage, read> times: array<u32>;

@group(0) @binding(2)
var<storage, read> specs: array<u32>;

@group(0) @binding(3)
var<storage, read> amps: array<f32>;

@group(0) @binding(4)
var<storage, read> periods: array<u32>;

@group(0) @binding(5)
var<storage, read> min_amps: array<f32>;

@group(0) @binding(6)
var<storage, read_write> hist: array<atomic<u32>>;

@group(0) @binding(7)
var<storage, read_write> total_count: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.n_events) {
        return;
    }

    let period = periods[idx];
    if (period == 0xFFFFFFFFu) {
        return;
    }

    let t = times[idx];
    let spec = specs[idx];
    let amp = amps[idx];

    if (t >= params.min_time && t < params.max_time && spec < params.n_spec) {
        if (amp > min_amps[spec]) {
            let bin = u32(f32(t - params.min_time) * params.inv_width);
            if (bin < params.n_bins) {
                let hist_idx = (period * params.n_spec + spec) * params.n_bins + bin;
                atomicAdd(&hist[hist_idx], 1u);
                atomicAdd(&total_count[0], 1u);
            }
        }
    }
}

