import time

from MNeuEventLib import Histogram, NexusData, Filters

files = [f"SIM0000000{n}.nxs" for n in range(1,4)]

stats = 1
n_filters = 2
n_spec = 960

def add_N_filters(data, N):
    """
    Simple method for adding N exclude filters,
    they are placed every other frame.
    This maximises the computational expense
    of the calculation.
    :param data: the NexusData object
    :param N: the number of filters
    :return: A Filters object with N time filters 
    """
    filters = Filters()
    filters.set_time_type("exclude")

    if N == 0:
        return filters
    frames = data.get_frame_times() * 1e-9
    offset = frames[100]
    m = 0
    skip = False
    for j in range(len(frames)-1):
        width = frames[j+1] - frames[j]
        if width > 0 and not skip:
            filters.add_time_filter(f'tmp_{m}',
                                    offset*(j+1) + frames[j] + .2*width,
                                    offset*(j+1) + frames[j] + 7.8*width)
            skip = True
            m += 1
        elif m == N:
            return filters 
        else:
            skip = False

    return filters

for file in files:
    print("\nFile: ", file)

    data = NexusData(file, 960)
    filters = add_N_filters(data, n_filters)

    avg_calc_time = 0
    avg_run_time = 0
    for _ in range(0, stats):
        start_time = time.time()
        histogram = Histogram(0., 32.768, 2048)
        result, calc_time = histogram.calculate(data, filters)
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

