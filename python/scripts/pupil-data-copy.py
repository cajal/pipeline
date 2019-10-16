from pipeline import pupil
import datajoint as dj

# Tracking

# tracking_old_table = pupil.ManuallyTrackedContours.proj()
# tracking_already_processed = (dj.U('animal_id', 'session', 'scan_idx') & (pupil.Tracking & 'tracking_method=1'))
# tracking_to_be_processed = tracking_old_table - tracking_already_processed

# Fitting
# Fitting_old_table = pupil.FittedContour.proj()
# Fitting_already_processed = (dj.U('animal_id', 'session', 'scan_idx') & (pupil.FittedPupil & 'tracking_method=1'))
# Fitting_to_be_processed = Fitting_old_table - Fitting_already_processed

# pupil.FittedPupil.populate(Fitting_to_be_processed, {'tracking_method':1},
#                         reserve_jobs=True, suppress_errors=True)

# fittin for zhiwei

processed = [{'animal_id': 17797, 'session': 6, 'scan_idx': 6},
 {'animal_id': 17797, 'session': 6, 'scan_idx': 7},
 {'animal_id': 17797, 'session': 7, 'scan_idx': 3},
 {'animal_id': 17797, 'session': 7, 'scan_idx': 4},
 {'animal_id': 17797, 'session': 7, 'scan_idx': 5},
 {'animal_id': 17797, 'session': 8, 'scan_idx': 2},
 {'animal_id': 17797, 'session': 8, 'scan_idx': 5},
 {'animal_id': 17797, 'session': 8, 'scan_idx': 7},
 {'animal_id': 17797, 'session': 8, 'scan_idx': 8},
 {'animal_id': 17797, 'session': 8, 'scan_idx': 9},
 {'animal_id': 17797, 'session': 9, 'scan_idx': 2},
 {'animal_id': 17797, 'session': 9, 'scan_idx': 3},
 {'animal_id': 17797, 'session': 9, 'scan_idx': 4},
 {'animal_id': 17797, 'session': 9, 'scan_idx': 6}]

plat = dj.create_virtual_module('pipeline_platinum','pipeline_platinum')
plat_scan = plat.Scan() & 'animal_id = 17797' & 'notes like "%%platinum%%"'
to_be_processed = (plat_scan - processed).proj().fetch(as_dict=True)

for scan in to_be_processed[3:]:
    pupil.FittedPupil.populate(dict(scan, tracking_method=2),
                               reserve_jobs=True, suppress_errors=True)