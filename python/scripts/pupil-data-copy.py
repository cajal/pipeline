from pipeline import pupil
import datajoint as dj

# Tracking

# tracking_old_table = pupil.ManuallyTrackedContours.proj()
# tracking_already_processed = (dj.U('animal_id', 'session', 'scan_idx') & (pupil.Tracking & 'tracking_method=1'))
# tracking_to_be_processed = tracking_old_table - tracking_already_processed

# Fitting
Fitting_old_table = pupil.FittedContour.proj()
Fitting_already_processed = (dj.U('animal_id', 'session', 'scan_idx') & (pupil.FittedPupil & 'tracking_method=1'))
Fitting_to_be_processed = Fitting_old_table - Fitting_already_processed

pupil.FittedPupil.populate(Fitting_to_be_processed, {'tracking_method':1},
                        reserve_jobs=True, suppress_errors=True)
