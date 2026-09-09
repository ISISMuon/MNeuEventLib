from mantid.simpleapi import (CreateWorkspace,
                              RenameWorkspace,
                              MuonPreProcess,
                              MuonGroupingCounts,
                              EstimateMuonAsymmetryFromCounts,
                              MuonPairingAsymmetry,
                              Minus,
                              Load,
                              LoadMuonNexusV2)


def mantid_workflow(loader):
    # load data
    file = '/mnt/babylon/Public/Anthony_Lim/HIFI42.nxs'
    (ws, _, T0, FG,
     LG, _, _, det,
     _, ws1, ws2) = Loader(Filename=file)
    periods = ['1', '2']
    
    # mantid expects specific names for the workspaces
    RenameWorkspace(InputWorkspace=ws1,
                    OutputWorkspace='HIFI207745_raw_data_period_1 MA')
    RenameWorkspace(InputWorkspace=ws2,
                    OutputWorkspace='HIFI207745_raw_data_period_2 MA')
    
    # pre-process step
    MPP = MuonPreProcess(InputWorkspace=ws,
                         TimeMin=0,
                         TimeOffset=T0,
                         DeadTimeTable=det)
    
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
    
 

def test_mantid_works():
    ws = CreateWorkspace([1, 2], [3, 4])


def test_mantid_workflow_Load():
    mantid_workflow(Load)


def test_mantid_workflow_LoadMuonNexusv2():
    mantid_workflow(LoadMuonNexusV2)

