from pipeline import pupil
import datajoint as dj

old_table = pupil.ManuallyTrackedContours.proj()
already_processed = (dj.U('animal_id', 'session', 'scan_idx') & (pupil.Tracking & 'tracking_method=1'))
to_be_processed = old_table - already_processed

# pupil.Tracking.populate(to_be_processed, {'tracking_method':1},
#                         reserve_jobs=True, suppress_errors=True)
pupil.FittedPupil.populate(to_be_processed, {'tracking_method':1},
                           reserve_jobs=True, suppress_errors=True)
