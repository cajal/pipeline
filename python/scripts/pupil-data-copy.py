from pipeline import pupil
import datajoint as dj

"""
    Copying the old tracking (pupil.ManuallyTrackedContours) and fitting (pupil.FittedContour) data 
    to the new tables (pupil.Tracking & pupil.FittedPupil)
    
"""
# Tracking
jobs = (pupil.Tracking & 'tracking_method=1').proj() - (pupil.FittedPupil & 'tracking_method=1').proj() 

# Fitting
Fitting_old_table = pupil.FittedContour.proj()
Fitting_already_processed = (dj.U('animal_id', 'session', 'scan_idx') & (pupil.FittedPupil & 'tracking_method=1'))
Fitting_to_be_processed = Fitting_old_table - Fitting_already_processed

pupil.Tracking.populate(jobs, {'tracking_method':1},
                        reserve_jobs=True, suppress_errors=True)
pupil.FittedPupil.populate(Fitting_to_be_processed, reserve_jobs=True, suppress_errors=True)
