#!/usr/local/bin/python3
from pipeline import experiment, reso, meso, fuse, stack, pupil, treadmill, posture
from stimulus import stimulus
from stimline import tune
import time
import logging
import datajoint as dj

## database logging code 

logging.basicConfig(level=logging.ERROR)
logging.getLogger('datajoint.connection').setLevel(logging.DEBUG)
if hasattr(dj.connection, 'query_log_max_length'):
    dj.connection.query_log_max_length = 3000 


# # Scans
# for priority in range(120, -130, -10):  # highest to lowest priority
#     next_scans = (experiment.AutoProcessing() & 'priority > {}'.format(priority) &
#                   (experiment.Scan() & 'scan_ts > "2019-01-01 00:00:00"'))

next_scans = (experiment.AutoProcessing  & 'priority < 120' &
              (experiment.Scan & 'scan_ts > "2019-01-01 00:00:00"'))

pupil.FittedPupil().populate(next_scans,reserve_jobs=True)
