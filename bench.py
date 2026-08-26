import time

from MNeuEventLib import Data

files = [f"../data/SIM0000000{n}.nxs" for n in range(1,4)]

stats = 1
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


for file in files:
    print("\nFile: ", file)

    data = Data(file, 960)
    data.set_time_type("exclude")
    add_N_filters(data, n_filters)

    avg_run_time = 0
    for _ in range(0, stats):
        start_time = time.time()
        result = data.calculate()
        n = result.get_n_events()
        duration = time.time() - start_time
        avg_run_time += duration
    avg_run_time /= stats
    print("  Average run time: ", avg_run_time * 1e3, " ms",
          "\n  Number of events:", n,
          "\n  Millions of events per second:", (n / avg_run_time) * 1e-6)

