#!/usr/local/bin/python3
from pipeline import pupil, experiment
import logging 
import time
import datajoint as dj

## database logging code 

logging.basicConfig(level=logging.ERROR)
logging.getLogger('datajoint.connection').setLevel(logging.DEBUG)
if hasattr(dj.connection, 'query_log_max_length'):
    dj.connection.query_log_max_length = 3000 

while True:

    # deeplabcut pupil tracking
    next_scans = experiment.MesoClosedLoop()
    pupil.Tracking.populate(next_scans, {'tracking_method': 2}, reserve_jobs=True, suppress_errors=True)
    
    time.sleep(60)