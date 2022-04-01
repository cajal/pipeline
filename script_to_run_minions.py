#!/usr/local/bin/python3
from pipeline import experiment, reso, meso, fuse, stack, pupil, treadmill, posture

next_scans = [{'animal_id':26872, 'session':17, 'scan_idx':20},
            {'animal_id':27204, 'session':5, 'scan_idx':13},
            {'animal_id':27203, 'session':5, 'scan_idx':17},
            ]

# reso/meso
for pipe in [reso, meso]:
    pipe.Activity.populate(next_scans, {'spike_method': 5}, reserve_jobs=True,
                        suppress_errors=True)
    full_scans = (pipe.ScanInfo.proj() & pipe.Activity) - (pipe.ScanInfo.Field -
                                                        pipe.Activity)
    pipe.ScanDone.populate(full_scans & next_scans, reserve_jobs=True,
                        suppress_errors=True)
 
fuse.Activity.populate(next_scans, reserve_jobs=True, suppress_errors=True)
fuse.ScanDone.populate(next_scans, reserve_jobs=True, suppress_errors=True)