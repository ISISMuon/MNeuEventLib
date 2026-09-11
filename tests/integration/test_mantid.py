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


def mantid_workflow(loader, num):
    """
    Creates a histogram file from event data, then executes a Mantid muon workflow.
    :param loader: the Mantid load method to use
    :param num: a number to make the files unique in case of multiple runs
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
    hist_file = os.path.join(dir_path, f'HIFI{num}.nxs')
    data.save(hist_file, autofill=True)

    # load data
    result = loader(Filename=hist_file, deadtimeTable='deadtimes')
    
    # unpack results (Load and LoadMuonNexusV2 has different
    # number of return values
    ws = result[0]
    T0 = result[2]
    FG = result[3]
    LG = result[4]
    DT = result[6]

    ws1 = ws[0]
    ws2 = ws[1]
    
    # mantid expects specific names for the workspaces
    RenameWorkspace(InputWorkspace=ws1,
                    OutputWorkspace='HIFI207745_raw_data_period_1 MA')
    RenameWorkspace(InputWorkspace=ws2,
                    OutputWorkspace='HIFI207745_raw_data_period_2 MA')
    
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
    
    # diff in pair asymmetries
    diff = Minus(LHSWorkspace='long1',
                 RHSWorkspace='long2')
    
    # clean up
    os.remove(hist_file)

def test_mantid_works():
    """
    Test that Mantid is properly installed with a simple workflow.
    """
    _ = CreateWorkspace([1, 2], [3, 4])


def test_mantid_workflow_Load():
    """
    Test that Mantid can load the library's output histogram data
    using the 'Load' method.
    The 'Load' method checks the file against all of the different
    load algoriths in Mantid and uses the one with the 'best' match.
    It should identify the file as a Muon Nexus V2.
    """
    mantid_workflow(Load, 42)


def test_mantid_workflow_LoadMuonNexusv2():
    """
    Test that Mantid can load the the library's output histogram data
    using the 'LoadMuonNexusV2' method.
    This method is what should be called by the 'Load' method
    and should be used when the
    file is known to be a Muon Nexus V2 file.
    """
    mantid_workflow(LoadMuonNexusV2, 51)

