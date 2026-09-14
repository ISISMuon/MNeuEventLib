import h5py
import numpy as np
import shutil


def replace(f, key, values, dtype):
    """
    A simple helper method to replace
    a dataset (without attributes)
    """
    del f[key]
    f.create_dataset(key, data=values, dtype=dtype)

def make_single_period_data(multi_period,
                            single_period):
    """
    This is a simple method to convert
    multi-period data into single 
    period.
    :param multi_period: the file path
    to the multi-period data.
    :param single_period: the file path
    to write the single period data to
    """
    shutil.copy(multi_period,
                single_period)
    with h5py.File(single_period, 'a') as f:
        data = f['raw_data_1']
        tmp = data['detector_1_events']
        N = tmp['period_number'].len()
        zeros = np.zeros(N)
        replace(tmp, 'period_number', zeros, 'int64')

        tmp = data['periods']
        replace(tmp, 'labels', 'period_1', 'S8')

        replace(tmp, 'number', 1, 'int32')
        replace(tmp, 'type', [1], 'int32')
    return
