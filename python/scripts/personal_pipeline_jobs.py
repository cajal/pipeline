import datajoint as dj
import numpy as np
from stimline import tune

"""
    Modify here if you wanna process your own jobs. Use this as a template
"""

#jake's diamond scans
session_list = [4,5,5,5,7,7,7,8,8,8,9,9,9]
scan_idx_list = [1,2,3,4,5,8,9,5,8,11,5,7,8]
animal_id_list = [21617] * len(session_list)



for _id, session, scan_idx in zip(animal_id_list, session_list, scan_idx_list):
    key = dict(animal_id=_id, session=session, scan_idx=scan_idx)

    tune.STA.populate(key, reserve_jobs=True, suppress_errors=True)
    tune.STAQual.populate(key, reserve_jobs=True, suppress_errors=True)
    tune.Ori(key, reserve_jobs=True, suppress_errors=True)
    tune.Kuiper(key, reserve_jobs=True, suppress_errors=True)
    tune.MovieOracle(key, reserve_jobs=True, suppress_errors=True)
