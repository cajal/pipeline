#!/usr/local/bin/python3
from pipeline import stack, pupil, experiment


# 3-d segmentation
stack.Segmentation.populate(reserve_jobs=True, suppress_errors=True)

# deeplabcut pupil tracking
next_scans = experiment.AutoProcessing() & (experiment.Scan() & 'scan_ts > "2019-01-01 00:00:00"')
pupil.Tracking.populate(next_scans, {'tracking_method': 2}, reserve_jobs=True,
                        suppress_errors=True)
