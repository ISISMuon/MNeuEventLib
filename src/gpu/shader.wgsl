struct Params {
    min_time: u32,
    max_time: u32,
    n_bins: u32,
    n_spec: u32,
    inv_width: f32,
    n_events: u32,
    baseline_min_amp: f32,
    flags: u32,
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

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;
    if (idx < params.n_events) {
        let is_uniform_period = (params.flags & 2u) != 0u;
        let period = select(periods[idx], 0u, is_uniform_period);
        if (period != 0xFFFFFFFFu) {
            let t = times[idx];
            if (t >= params.min_time && t < params.max_time) {
                let spec = specs[idx];
                if (spec < params.n_spec) {
                    let amp = amps[idx];
                    let is_uniform_amp = (params.flags & 1u) != 0u;
                    let min_amp = select(min_amps[spec], params.baseline_min_amp, is_uniform_amp);
                    if (amp > min_amp) {
                        let bin = u32(f32(t - params.min_time) * params.inv_width);
                        if (bin < params.n_bins) {
                            let hist_idx = (period * params.n_spec + spec) * params.n_bins + bin;
                            atomicAdd(&hist[hist_idx], 1u);
                        }
                    }
                }
            }
        }
    }
}
