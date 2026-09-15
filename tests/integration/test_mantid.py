import os
from mantid.simpleapi import (CreateWorkspace,
                              RenameWorkspace,
                              MuonPreProcess,
                              MuonGroupingCounts,
                              EstimateMuonAsymmetryFromCounts,
                              MuonPairingAsymmetry,
                              Minus,
                              Load,
                              LoadMuonNexusV2)
from MNeuEventLib import Data
from MNeuEventLib.test_helpers import make_single_period_data


def mantid_workflow(load_result, periods):
    """
    Executes a Mantid muon workflow from the result of a loader.
    :param load_result: the result of Mantid load method
    :param periods: a list of the periods for the dataset
    """
   
    # unpack results (Load and LoadMuonNexusV2 has different
    # number of return values
    ws = load_result[0]
    T0 = load_result[2]
    FG = load_result[3]
    LG = load_result[4]
    DT = load_result[6]

    if len(periods) > 1:
        ws1 = ws[0]
        ws2 = ws[1]
        # mantid expects specific names for the workspaces
        RenameWorkspace(InputWorkspace=ws1,
                        OutputWorkspace='HIFI207745_raw_data_period_1 MA')
        RenameWorkspace(InputWorkspace=ws2,
                        OutputWorkspace='HIFI207745_raw_data_period_2 MA')
    else:
        # mantid expects specific names for the workspaces
        RenameWorkspace(InputWorkspace=ws,
                        OutputWorkspace='HIFI207745_raw_data MA')
    
    # pre-process step
    MPP = MuonPreProcess(InputWorkspace=ws,
                         TimeMin=0,
                         TimeOffset=T0,
                         DeadTimeTable=DT)
    
    # counts for the groups and periods
    groups = {'fwd': '1-32', 'bwd': '33-64'}
    counts = {}
    for p in periods:
        for g in groups.keys():
            name = g + p
            counts[name] = MuonGroupingCounts(InputWorkspace=MPP,
                                              OutputWorkspace=name+'_counts',
                                              GroupName=name,
                                              Grouping=groups[g],
                                              summedPeriods=p)
    
    # asymmetry for the groups and periods
    group_asym = {}
    for g in counts.keys():
        name = g+'_Asymmetry',
        group_asym[name] = EstimateMuonAsymmetryFromCounts(InputWorkspace=counts[g],
                                                           OutputWorkspace=g+'_Asymmetry',
                                                           OutputUnNormData=False,
                                                           StartX=FG,
                                                           EndX=LG)
    # pair asymmetry for the periods
    for p in periods:
        MuonPairingAsymmetry(OutputWorkspace='long'+p,
                             PairName='long'+p,
                             InputWorkspace1=counts['fwd'+p],
                             InputWorkspace2=counts['bwd'+p])
    
    if len(periods) > 1:
        # diff in pair asymmetries
        diff = Minus(LHSWorkspace='long1',
                     RHSWorkspace='long2')
        

def test_mantid_works():
    """
    Test that Mantid is properly installed with a simple workflow.
    """
    _ = CreateWorkspace([1, 2], [3, 4])


def test_mantid_workflow_Load_multi():
    """
    Test that Mantid can load the library's output histogram data
    using the 'Load' method for multi-period data.
    The 'Load' method checks the file against all of the different
    load algoriths in Mantid and uses the one with the 'best' match.
    It should identify the file as a Muon Nexus V2.
    """
    periods = ['1', '2']
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file = os.path.join(dir_path,
                        '..',
                        'test_data',
                        'HIFI00195790.nxs')
   
    # create histogram data from events
    data = Data(file, 64)
    result = data.calculate()
    hist_file = os.path.join(dir_path, f'HIFI002.nxs')
    data.save(hist_file, autofill=True)

    # mantid workflow
    result = Load(Filename=hist_file, deadtimeTable='deadtimes')
 
    mantid_workflow(result, periods)
    os.remove(hist_file)

def test_mantid_workflow_LoadMuonNexusv2_multi():
    """
    Test that Mantid can load the the library's output histogram data
    using the 'LoadMuonNexusV2' method for multiperiod data.
    This method is what should be called by the 'Load' method
    and should be used when the
    file is known to be a Muon Nexus V2 file.
    """
    periods = ['1', '2']
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file = os.path.join(dir_path,
                        '..',
                        'test_data',
                        'HIFI00195790.nxs')
   
    # create histogram data from events
    data = Data(file, 64)
    result = data.calculate()
    hist_file = os.path.join(dir_path, f'HIFI021.nxs')
    data.save(hist_file, autofill=True)

    # mantid workflow
    result = LoadMuonNexusV2(Filename=hist_file, deadtimeTable='deadtimes')
 
    mantid_workflow(result, periods)
    os.remove(hist_file)

def test_mantid_workflow_Load_single():
    """
    Test that Mantid can load the library's output histogram data
    using the 'Load' method for single-period data.
    The 'Load' method checks the file against all of the different
    load algoriths in Mantid and uses the one with the 'best' match.
    It should identify the file as a Muon Nexus V2.
    """
    periods = ['1']
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file = os.path.join(dir_path,
                        '..',
                        'test_data',
                        'HIFI00195790.nxs')
    event_file = os.path.join(dir_path, f'HIFI001.nxs')
    make_single_period_data(file, event_file)
    # create histogram data from events
    data = Data(event_file, 64)
    result = data.calculate()
    hist_file = os.path.join(dir_path, f'HIFI0032.nxs')
    data.save(hist_file, autofill=True)
    del data  # release file handle before removal (Windows WinError 32)

    # mantid workflow
    result = Load(Filename=hist_file, deadtimeTable='deadtimes')
 
    mantid_workflow(result, periods)
    os.remove(hist_file)
    os.remove(event_file)

def test_mantid_workflow_LoadMuonNexusv2_single():
    """
    Test that Mantid can load the the library's output histogram data
    using the 'LoadMuonNexusV2' method for multiperiod data.
    This method is what should be called by the 'Load' method
    and should be used when the
    file is known to be a Muon Nexus V2 file.
    """
    periods = ['1']
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file = os.path.join(dir_path,
                        '..',
                        'test_data',
                        'HIFI00195790.nxs')
   
    event_file = os.path.join(dir_path, f'HIFI0007.nxs')
    make_single_period_data(file, event_file)
    # create histogram data from events
    data = Data(event_file, 64)
    result = data.calculate()
    hist_file = os.path.join(dir_path, f'HIFI042.nxs')
    data.save(hist_file, autofill=True)
    del data  # release file handle before removal (Windows WinError 32)

    # mantid workflow
    result = LoadMuonNexusV2(Filename=hist_file, deadtimeTable='deadtimes')
 
    mantid_workflow(result, periods)
    os.remove(hist_file)
    os.remove(event_file)
