#!/usr/local/bin/python3
import kubernetes
import time
import datajoint as dj

experiment = dj.create_virtual_module('experiment', 'pipeline_experiment')
treadmill = dj.create_virtual_module('treadmill', 'pipeline_treadmill')
pupil = dj.create_virtual_module('pupil', 'pipeline_pupil')
fuse = dj.create_virtual_module('fuse', 'pipeline_fuse')


while True:

    unfinished_treadmill = experiment.MesoClosedLoop - treadmill.Treadmill
    unfinished_pupil = experiment.MesoClosedLoop - pupil.Eye
    unfinished_fuse = experiment.MesoClosedLoop - fuse.ScanDone

    if unfinished_treadmill or unfinished_pupil or unfinished_fuse:
        pass

        '''
        TODO: add meso_closed_loop=true:NoExecute taint to the following nodes:
            at-compute003
            at-compute004
            at-compute005
        '''
        
    else:
        pass

        '''
        TODO: remove meso_closed_loop=true:NoExecute taint from the following nodes:
            at-compute003
            at-compute004
            at-compute005
        '''

    time.sleep(60)