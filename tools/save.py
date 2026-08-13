import h5py
import os
import sys
import shutil
import copy
import os
from MNeuEventLib import Data
import numpy as np

def set_attributes(obj, dest):
    for attr in obj.attrs:
        if attr not in dest.attrs.keys():
            dest.attrs.create(attr, obj.attrs[attr])

def set_defaults(source, dest, name=None):
    # Retrieve the object being copied
    if isinstance(source, h5py.HLObject):
        obj = source
    else:
        obj = self[source]

    obj_name = obj.name.split('/')[-1].split('dataset_')[-1]
    dest_keys = dest.keys()
    # work out if the item already exists
    if isinstance(obj, h5py.Group):
        tmp = None
        if obj_name not in dest_keys and 'dataset_' not in obj.name:
            print('Copy group', obj_name)
            tmp = dest.require_group(obj.name)
            set_attributes(obj, tmp)
        elif obj_name not in dest_keys:
            # create default dataset
            print('Create default', obj_name)
            tmp = dest.create_dataset(obj.name.split('dataset_')[1], shape=shapes[obj['shape'][()].decode()], dtype=obj['dtype'][()], fillvalue=obj['default'][()])
            set_attributes(obj, tmp)
            return
        elif isinstance(dest[obj_name], h5py.Dataset):
            set_attributes(obj, dest[obj_name])
            return
        else:
            # if group exists just check the attributes are complete
            tmp = dest[obj.name]
            set_attributes(obj, dest)

        for key in obj.keys():
            set_defaults(obj[key], tmp, key)
 
    
    elif isinstance(obj, h5py.Dataset):
        if obj_name in dest.keys():
            # already exists
            set_attributes(obj, dest[obj_name])
            return
        else:
            print("Copy dataset", obj_name, f'##{obj[()]}##')
            dest.create_dataset(obj.name, data=obj[()], shape=obj.shape, dtype=str(obj.dtype))
            set_attributes(obj, dest[obj_name])
            return
 
def clean_up(new_file):

        #del new_file['raw_data_1/IDF_version']
        #new_file.create_dataset('raw_data_1/IDF_version', data=[2])
        
        # alpha is 1 not 0
        
        #new_file['raw_data_1/instrument/detector_1'].attrs['NX_class'] = 'NXdetector'
        #del new_file['raw_data_1/detector_1']
        
        #det = new_file.require_group('raw_data_1/detector_1')

        #ref = new_file['raw_data_1/instrument/detector_1']
        #for key in ref.keys():
            
        #    data = det.create_dataset(key, data=ref[key][()], shape=ref[key].shape, dtype=str(ref[key].dtype))
        #    set_attributes(ref[key], data)
        #det.create_dataset('period_index', data=[1, 2])
        #del new_file['raw_data_1/experiment_identifier'] 
        #new_file.create_dataset('raw_data_1/experiment_identifier', data=[2432])

        #del new_file['raw_data_1/periods/number']
        #new_file.create_dataset('raw_data_1/periods/number', [2])

        #det.attrs['NX_class'] = "NXdata"

        #for tmp in [det, new_file['raw_data_1/instrument/detector_1']]:
        #    d = tmp['raw_time']
        #    tmp.create_dataset('corrected_time', data = (d[1:] - d[:-1])/2.)
        #    tmp['corrected_time'].attrs['axis'] = "1"


        #    tmp = tmp['counts']
        #    tmp.attrs['first_good_bin'] = '8'
        #    tmp.attrs['last_good_bin'] = '2048'
        #    tmp.attrs['long_name'] = "positron_counts"
        #    tmp.attrs['offset'] = 8000
        #    #tmp.attrs['signal'] = "1"
        #    tmp.attrs['t0_bin'] = 2
        #    tmp.attrs['units'] = "counts"
        #    tmp.attrs['target'] = "/raw_data_1/instrument/detector_1/counts"

        tmp = new_file['raw_data_1/periods']
        tmp.create_dataset('good_frames', data=[999, 888], dtype='int32')
        
        #tmp = new_file['raw_data_1/good_frames'][()]
        #del new_file['raw_data_1/good_frames']
        #new_file.create_dataset('raw_data_1/good_frames', data=[tmp])

        new_file['raw_data_1/detector_1/counts'].attrs['first_good_bin'] = '1'
        new_file['raw_data_1/detector_1/counts'].attrs['last_good_bin'] = '1000'
        del new_file['raw_data_1/name']
        new_file.create_dataset('raw_data_1/name', data='HIFI')

        tmp = new_file['raw_data_1/sample']
        # maybe we delete it and add it back in?
        for name in tmp.keys():
            #print('moo', name, 'object'== str(tmp[name].dtype), 'boo')
            #if name.split('/')[-1] not in ['name', 'probe', 'type']:
            if isinstance(tmp[name], h5py.Group):
                continue
            else:
                if name == 'thickness':
                    
                    att = copy.copy(tmp[name].attrs)
                    del tmp[name]
                    tmp.create_dataset(name, data=0, dtype='float32')
                    for a in att.keys():
                        tmp.attrs[a] = att[a]
                    
                elif name in ['type', 'description']:
                    data = copy.copy(tmp[name][()])
                    att = copy.copy(tmp[name].attrs)
                    del tmp[name]
                    tmp.create_dataset(name, data=data)
                    for a in att.keys():
                        tmp.attrs[a] = att[a]
 
        tmp = new_file['raw_data_1/instrument/source']
        # maybe we delete it and add it back in?
        for name in tmp.keys():
            print('moo', name, tmp[name][()], 'boo')
            if name in ['muon_energy', 'muon_momentum', 'muon_pulse_width', 'name', 'notes', 'pion_momentum', 'probe', 'source_current', 'source_energy', 'source_frequency', 'source_pulse_width', 'target_material', 'target_thickness', 'type']:
                print('yay')
                data = copy.copy(tmp[name][()])
                att = copy.copy(tmp[name].attrs)
                del tmp[name]
                tmp.create_dataset(name, data=[data])
                for a in att.keys():
                    tmp.attrs[a] = att[a]
        #    print('check', name, tmp[name][()])

def save_default(file_name):
    with h5py.File('REF_file.nxs', 'r') as file:
        with h5py.File(file_name, 'a') as new_file:
            clean_up(new_file)
            # do top level manually
            for key in file.keys():
                new_file.require_group(key)
                # add attributes
                for attr in file[key].attrs:
                    new_file[key].attrs.create(attr, file[key].attrs[attr])
                new_keys = new_file.keys()
                for tmp in file[key].keys():
                    if tmp=='selog':
                        print('skip')
                    elif tmp not in new_keys:
                        # copies the dataset or (group and contents)
                        set_defaults(file[key][tmp], new_file[key])

            #del new_file['raw_data_1/runlog/good_frames/time']
            #del new_file['raw_data_1/runlog/good_frames/value']

            #new_file.create_dataset('raw_data_1/runlog/good_frames/time', data=[0,1])
            #new_file.create_dataset('raw_data_1/runlog/good_frames/value', data=[10, 81630])
            
            #new_file.create_dataset('raw_data_1/periods/good_frames_daq', data=[27210, 27210, 27210])
        


def get_p_info(file_name):
    with h5py.File(file_name,'r') as file:
        labels = file['raw_data_1/periods/labels'][()].decode()

    return len(labels.split(',')), 0


#input_file = 'HIFI00209192_events.nxs'
input_file = 'HIFI00207745_events.nxs'
#input_file = 'HIFI00200977_events.nxs'
#input_file = os.path.join('..', '..', 'save_test', 'HIFI00206202_events.nxs')
#input_file = 'ZOOM00040384.nxs'

#input_file = os.path.join('..', '..', 'data_cf', 'events', 'HIFI00209192.nxs')

N = 64
output_file = 'HIFI51.nxs'
tmp_file = 'HIFI007.nxs'

data = Data(input_file, N)#98304)#64)
#data.add_time_filter('time', 0, 20)
data.calculate()
data.save(tmp_file)

shutil.copyfile(tmp_file, output_file)

######## need to get this from file ############
periods, dwell = get_p_info(input_file)
print(periods, 'periods')
#periods = 3
#dwell = 0
shapes = {'N': N,
          'P': periods}
shapes['NP'] = shapes['N']*shapes['P']
shapes['PD'] = shapes['P'] + dwell
shapes['NPD'] = shapes['N']*shapes['PD']

save_default(output_file)

print('done!!!')

