"""
This script can be used to generate a reference file.
The reference file is then used for filling out an
incomplete event file so it can be loaded into Mantid.
To use this you need an example histogram file with
the correct format. 

If the reference file is saved as <ref> (assume 
this includes the file path), then when saving 
event data to histograms, the command would be

data.save(<output>, autofill=Ture, ref_file=<ref>)

where <output> is the name of the file you want
to save to, and data is the event data 
object which would have been loaded from a 
events data file.
"""

import h5py

"""
This data needs to be provided from the event 
file. These datasets are calculated as part
of the histogramming process.
"""
skip = ['collection_time',
        'corrected_time',
        'counts',
        'period_index',
        'raw_time',
        'spectrum_index',
        'duration',
        'end_time',
        'good_frames',
        'resolution',
        'frames_requested',
        'good_frames',
        'good_frames_daq',
        'raw_frames',
        'start_time',
        ]

"""
This is the period information for the example
histogram file. It is recommended that it includes
multiple periods and Dwell. 
N: number of detectors
P: number of periods
PD: number of periods + Dwell
NP: number of detectors times number of periods
NPD: number of detectors times the number of periods and Dwell
"""
N = 64
P = 2
PD = 3
NP = N*P
NPD = N*PD

"""
These strings are copied from the
histogram file to the default file
"""
keep_strings = ['definition',
                'probe',
                'type',
                ]
          
def set_attributes(obj, dest):
    """
    Copy attributes from one object to another.
    :param obj: the example file's open object
    :param dest: the ref file's open object
    """
    for attr in obj.attrs:
        dest.attrs.create(attr, obj.attrs[attr])
        
def read(obj, new_obj, key):
    """
    Recursive copy of data, replacing any dataset
    with a "default dataset" if the length
    depends on the number of detectors, periods,
    or Dwell. Otherwise it records the dataset values
    as 0.
    :param obj: the example file's open object
    :param new_obj: the ref file's open object
    :param key: the name of the dataset
    """
    if isinstance(obj, h5py.Group):
        print('group', key)
        tmp = new_obj.require_group(key)
        set_attributes(obj, tmp)
        for new_key in obj.keys():
            read(obj[new_key], tmp, new_key)
        print('exit group')
        print()

    elif isinstance(obj, h5py.Dataset):
        name = obj.name.split('/')[-1]
        dtype = str(obj.dtype)
        shape = obj.shape
        val = obj[()]
        if 'float' in dtype or 'int' in dtype:
            if len(val) in [N, P, NP, PD, NPD]:
                length = 'N'
                if len(val) == P:
                    length = 'P'
                elif len(val) == NP:
                    length = 'NP'
                elif len(val) == PD:
                    length = 'PD'
                elif len(val) == NPD:
                    length = 'NPD'
                group = new_obj.require_group('dataset_' + name)
                set_attributes(obj, group)
                group.create_dataset('default', data=0)
                group.create_dataset('dtype', data=dtype)
                group.create_dataset('shape', data=length)

            else:
                tmp = new_obj.create_dataset(name,
                                             shape=shape,
                                             dtype=dtype,
                                             fillvalue=0)

                set_attributes(obj, tmp)
        else: # assume a string
            val = obj[()]
            tmp = None
            if name in keep_strings or (isinstance(val, str) and val=='ISIS'):
                tmp = new_obj.create_dataset(name, data=val, dtype=dtype)
            elif '.dat' in name:
                tmp = new_obj.create_dataset(name, data=val, dtype=dtype)
            elif obj.shape == [1]:
                tmp = new_obj.create_dataset(name, data=[' '], dtype='S1')
            else:
                tmp = new_obj.create_dataset(name, data=val, dtype=dtype)
            set_attributes(obj, tmp)


with h5py.File('HIFI00207745.nxs', 'r') as file:
    with h5py.File('REF_file.nxs', 'w') as new_file:
        # do top level manually
        for key in file.keys():
            new_obj = new_file.require_group(key)
            set_attributes(file[key], new_obj)
            for tmp in file[key].keys():
                if tmp in ['selog']:
                    print('skip')
                else:
                    read(file[key][tmp], new_obj, tmp)
                if tmp == 'detector_1':
                    new_obj[tmp].attrs['NX_class'] = 'NXdata'
# clean up
print('done')
 
