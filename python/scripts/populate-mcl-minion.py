#!/usr/local/bin/python3
from pipeline import experiment, meso, fuse, pupil, treadmill
from stimulus import stimulus
import time
import logging
import datajoint as dj

## database logging 
logging.basicConfig(level=logging.ERROR)
logging.getLogger('datajoint.connection').setLevel(logging.DEBUG)
if hasattr(dj.connection, 'query_log_max_length'):
    dj.connection.query_log_max_length = 3000 

next_scans = experiment.MesoClosedLoop

# stimulus
stimulus.Sync.populate(next_scans, reserve_jobs=True, suppress_errors=True)
stimulus.BehaviorSync.populate(next_scans, reserve_jobs=True, suppress_errors=True)

# treadmill, pupil
treadmill.Treadmill.populate(next_scans, reserve_jobs=True, suppress_errors=True)
pupil.Eye.populate(next_scans, reserve_jobs=True, suppress_errors=True)

# meso
meso.ScanInfo.populate(next_scans, reserve_jobs=True, suppress_errors=True)
meso.Quality.populate(next_scans, reserve_jobs=True, suppress_errors=True)
meso.RasterCorrection.populate(next_scans, reserve_jobs=True, suppress_errors=True)
meso.MotionCorrection.populate(next_scans, reserve_jobs=True, suppress_errors=True)
meso.SummaryImages.populate(next_scans, reserve_jobs=True, suppress_errors=True)
meso.Segmentation.populate(next_scans, reserve_jobs=True, suppress_errors=True)
meso.Fluorescence.populate(next_scans, reserve_jobs=True, suppress_errors=True)
meso.MaskClassification.populate(next_scans, {'classification_method': 2}, reserve_jobs=True, suppress_errors=True)
meso.ScanSet.populate(next_scans, reserve_jobs=True, suppress_errors=True)
time.sleep(60)
meso.Activity.populate(next_scans, reserve_jobs=True, suppress_errors=True)
full_scans = (meso.ScanInfo.proj() & meso.Activity) - (meso.ScanInfo.Field - meso.Activity)
meso.ScanDone.populate(full_scans & next_scans, reserve_jobs=True, suppress_errors=True)

# fuse
fuse.MotionCorrection.populate(next_scans, reserve_jobs=True, suppress_errors=True)
fuse.ScanSet.populate(next_scans, reserve_jobs=True, suppress_errors=True)
fuse.Activity.populate(next_scans, reserve_jobs=True, suppress_errors=True)
fuse.ScanDone.populate(next_scans, reserve_jobs=True, suppress_errors=True)
