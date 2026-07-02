import time

from MNeuEventLib import Histogram, NexusData 

files = [f"SIM0000000{n}.nxs" for n in range(1,4)]

stats = 1
n_spec = 960

for file in files:
    print("\nFile: ", file)

    data = NexusData(file, 960)

    avg_calc_time = 0
    avg_run_time = 0
    for _ in range(0, stats):
        start_time = time.time()
        histogram = Histogram(0., 32.768, 2048)
        result, calc_time = histogram.calculate(data)
        n = result.n_events()
        duration = time.time() - start_time
        avg_run_time += duration
        avg_calc_time += calc_time 
    avg_run_time /= stats
    avg_calc_time /= stats
    print("  Average run time: ", avg_run_time * 1e3, " ms",
          "\n  Average calc time: ", avg_calc_time, " ms",
          "\n  Number of events:", n,
          "\n  Millions of events per second:", (n / avg_run_time) * 1e-6)

