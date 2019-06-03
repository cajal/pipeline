#!/usr/local/bin/python3
import time

from pipeline import stack, pupil, experiment

while True:
    stack.Segmentation.populate(reserve_jobs=True, suppress_errors=True)

    next_scans = experiment.AutoProcessing() & (experiment.Scan() & 'scan_ts > "2019-01-01 00:00:00"')
    pupil.Tracking.populate(next_scans, {'tracking_method': 2},
                            reserve_jobs=True, suppress_errors=True)

    time.sleep(600)  # wait 10 minutes before checking again
