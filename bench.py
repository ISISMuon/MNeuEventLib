import time
import numpy as np
import MNeuEventLib as mel
from MNeuEventLib import Data

files = ["tools/HIFI00207745_events.nxs"]

stats = 10
n_filters = 2
n_spec = 960

def add_N_filters(data, N):
    """
    Simple method for adding N exclude filters,
    they are placed every other frame.
    This maximises the computational expense
    of the calculation.
    :param data: the Data object
    :param N: the number of filters 
    """
    if N == 0:
        return
    frames = data.dataset.get_frame_times() * 1e-9
    offset = frames[100]
    m = 0
    skip = False
    for j in range(len(frames)-1):
        width = frames[j+1] - frames[j]
        if width > 0 and not skip:
            data.add_time_filter(f'tmp_{m}',
                                 offset*(j+1) + frames[j] + .2*width,
                                 offset*(j+1) + frames[j] + 7.8*width)
            skip = True
            m += 1
        elif m == N:
            return 
        else:
            skip = False

print("=== MNeuEventLib Performance Benchmark ===")
gpu_avail = mel.is_gpu_available()
print(f"GPU Available: {gpu_avail}")
if gpu_avail:
    dev_info = mel.get_device_info()
    print(f"GPU Device Info: {dev_info}")

for file in files:
    print(f"\nBenchmark File: {file}")

    data = Data(file, n_spec)
    data.set_time_type("exclude")
    add_N_filters(data, n_filters)

    # 1. Parity Check
    print("\n--- Verifying Numerical Parity ---")
    data.invalidate_cache()
    res_cpu = data.calculate("cpu")
    hist_cpu = res_cpu.get_histogram()
    n_cpu = res_cpu.get_n_events()

    if gpu_avail:
        data.invalidate_cache()
        res_gpu = data.calculate("gpu")
        hist_gpu = res_gpu.get_histogram()
        n_gpu = res_gpu.get_n_events()

        data.invalidate_cache()
        res_hyb = data.calculate("hybrid")
        hist_hyb = res_hyb.get_histogram()
        n_hyb = res_hyb.get_n_events()

        assert n_cpu == n_gpu, f"GPU event count mismatch: {n_cpu} vs {n_gpu}"
        assert n_cpu == n_hyb, f"Hybrid event count mismatch: {n_cpu} vs {n_hyb}"
        np.testing.assert_array_equal(hist_cpu, hist_gpu, err_msg="GPU histogram mismatch with CPU!")
        np.testing.assert_array_equal(hist_cpu, hist_hyb, err_msg="Hybrid histogram mismatch with CPU!")
        print("PASS: CPU, GPU, and Hybrid produced IDENTICAL histograms and event counts!")
    else:
        print("GPU not available on this system; skipping GPU parity check.")

    # 2. Benchmark modes
    modes = ["cpu"]
    if gpu_avail:
        modes.extend(["gpu", "hybrid", "auto"])

    print(f"\n--- Running Benchmarks ({stats} iterations each) ---")
    results = {}
    for mode in modes:
        # Warmup
        data.invalidate_cache()
        data.calculate(mode)

        total_time = 0.0
        for _ in range(stats):
            data.invalidate_cache()
            t0 = time.perf_counter()
            res = data.calculate(mode)
            total_time += time.perf_counter() - t0

        avg_ms = (total_time / stats) * 1000.0
        n_ev = res.get_n_events()
        m_ev_per_sec = (n_ev / (total_time / stats)) * 1e-6
        results[mode] = (avg_ms, m_ev_per_sec)

        print(f"  [{mode:>6}]: {avg_ms:7.2f} ms | {m_ev_per_sec:6.2f} M events/s ({n_ev:,} events)")

    cpu_ms = results["cpu"][0]
    print("\n--- Summary & Relative Speedup vs CPU ---")
    for mode, (ms, m_ev) in results.items():
        speedup = cpu_ms / ms
        print(f"  {mode:>6}: {speedup:5.2f}x ({ms:.2f} ms)")

if hasattr(mel, "cleanup_gpu"):
    mel.cleanup_gpu()

import os
os._exit(0)

