import h5py

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

N = 64
P = 2
PD = 3
NP = N*P
NPD = N*PD
one = 1

keep_strings = ['definition',
                #'name', # should say ISIS
                'probe',
                'type',
                ]
          
def set_attributes(obj, dest):
    for attr in obj.attrs:
        dest.attrs.create(attr, obj.attrs[attr])
        
def read(obj, new_obj, key):
    # Get its value (if dataset)
    if isinstance(obj, h5py.Group):
        print('group', key)
        tmp = new_obj.require_group(key)
        set_attributes(obj, tmp)
        for new_key in obj.keys():
            read(obj[new_key], tmp, new_key)
        print('exit group')
        print()

    elif isinstance(obj, h5py.Dataset):# and obj.name.split('/')[-1] not in skip:
        name = obj.name.split('/')[-1]
        dtype = str(obj.dtype)
        shape = obj.shape
        val = obj[()]
        print("boo", name, dtype)
        if 'float' in dtype or 'int' in dtype:
            print('moo', len(val))
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
        #    print('')
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
                # if we need to update the group name
                if tmp in ['fdsafdsadsf']:#'selog']:
                    print('skip')
                else:
                #    #print('read', 
                    read(file[key][tmp], new_obj, tmp)
                #    # copies the group and contents
                if tmp == 'detector_1':
                    new_obj[tmp].attrs['NX_class'] = 'NXdata'
# clean up
print('done')
 
