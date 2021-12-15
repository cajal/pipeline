#!/usr/local/bin/python3
from pipeline import experiment, pupil
import logging
import datajoint as dj

## database logging code 
logging.basicConfig(level=logging.ERROR)
logging.getLogger('datajoint.connection').setLevel(logging.DEBUG)
if hasattr(dj.connection, 'query_log_max_length'):
    dj.connection.query_log_max_length = 3000 

next_scans = experiment.MesoClosedLoop
pupil.FittedPupil().populate(next_scans, reserve_jobs=True)
