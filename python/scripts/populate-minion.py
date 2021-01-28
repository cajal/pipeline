#!/usr/local/bin/python3
from pipeline import experiment, reso, meso, fuse, stack, pupil, treadmill, posture
from stimulus import stimulus
from stimline import tune
import time

# # Scans
# for priority in range(120, -130, -10):  # highest to lowest priority
#     next_scans = (experiment.AutoProcessing() & 'priority > {}'.format(priority) &
#                   (experiment.Scan() & 'scan_ts > "2019-01-01 00:00:00"'))

next_scans = (experiment.AutoProcessing  & 'priority < 120' &
              (experiment.Scan & 'scan_ts > "2019-01-01 00:00:00"'))

# stimulus
stimulus.Sync.populate(next_scans, reserve_jobs=True, suppress_errors=True)
stimulus.BehaviorSync.populate(next_scans, reserve_jobs=True, suppress_errors=True)

# treadmill, pupil, posture
treadmill.Treadmill.populate(next_scans, reserve_jobs=True, suppress_errors=True)
pupil.Eye.populate(next_scans, reserve_jobs=True, suppress_errors=True)
pupil.FittedPupil.populate(next_scans, reserve_jobs=True, suppress_errors=True)
posture.Posture.populate(next_scans, reserve_jobs=True, suppress_errors=True)

# stack
stack.StackInfo.populate(stack.CorrectionChannel, reserve_jobs=True, suppress_errors=True)
stack.Quality.populate(reserve_jobs=True, suppress_errors=True)
stack.RasterCorrection.populate(reserve_jobs=True, suppress_errors=True)
stack.MotionCorrection.populate(reserve_jobs=True, suppress_errors=True)
stack.Stitching.populate(reserve_jobs=True, suppress_errors=True)
stack.CorrectedStack.populate(reserve_jobs=True, suppress_errors=True)

# reso/meso
for pipe in [reso, meso]:
    pipe.ScanInfo.populate(next_scans, reserve_jobs=True, suppress_errors=True)
    pipe.Quality.populate(next_scans, reserve_jobs=True, suppress_errors=True)
    pipe.RasterCorrection.populate(next_scans, reserve_jobs=True, suppress_errors=True)
    pipe.MotionCorrection.populate(next_scans, reserve_jobs=True, suppress_errors=True)
    pipe.SummaryImages.populate(next_scans, reserve_jobs=True, suppress_errors=True)
    pipe.Segmentation.populate(next_scans, reserve_jobs=True, suppress_errors=True)
    pipe.Fluorescence.populate(next_scans, reserve_jobs=True, suppress_errors=True)
    pipe.MaskClassification.populate(next_scans, {'classification_method': 2},
                                     reserve_jobs=True, suppress_errors=True)
    pipe.ScanSet.populate(next_scans, reserve_jobs=True, suppress_errors=True)
    time.sleep(60)
    pipe.Activity.populate(next_scans, {'spike_method': 5}, reserve_jobs=True,
                           suppress_errors=True)
    full_scans = (pipe.ScanInfo.proj() & pipe.Activity) - (pipe.ScanInfo.Field -
                                                           pipe.Activity)
    pipe.ScanDone.populate(full_scans & next_scans, reserve_jobs=True,
                           suppress_errors=True)

# fuse
fuse.MotionCorrection.populate(next_scans, reserve_jobs=True, suppress_errors=True)
fuse.ScanSet.populate(next_scans, reserve_jobs=True, suppress_errors=True)
fuse.Activity.populate(next_scans, reserve_jobs=True, suppress_errors=True)
fuse.ScanDone.populate(next_scans, reserve_jobs=True, suppress_errors=True)

# more stack (needs corrected fields)
stack.PreprocessedStack.populate(stack.SegmentationTask, reserve_jobs=True,
                                 suppress_errors=True)
stack.FieldSegmentation.populate(reserve_jobs=True, suppress_errors=True)
stack.PreprocessedStack.populate(stack.RegistrationTask.proj(session='stack_session',
                                                             channel='stack_channel'),
                                 reserve_jobs=True, suppress_errors=True)
stack.Registration.populate(reserve_jobs=True, suppress_errors=True)

# # tune (these are memory intensive)
# tune.STA.populate(next_scans, reserve_jobs=True, suppress_errors=True)
# tune.STAQual.populate(next_scans, reserve_jobs=True, suppress_errors=True)
# tune.STAExtent.populate(next_scans, reserve_jobs=True, suppress_errors=True)
#
# tune.CaMovie.populate(next_scans, reserve_jobs=True, suppress_errors=True)
# tune.Drift.populate(next_scans, reserve_jobs=True, suppress_errors=True)
# tune.OriDesign.populate(next_scans, reserve_jobs=True, suppress_errors=True)
# tune.OriMap.populate(next_scans, reserve_jobs=True, suppress_errors=True)
# tune.Cos2Map.populate(next_scans, reserve_jobs=True, suppress_errors=True)
# tune.OriMapQuality.populate(next_scans, reserve_jobs=True, suppress_errors=True)
#
# # tune.OracleMap.populate(next_scans, reserve_jobs=True, suppress_errors=True)
# # tune.MovieOracle.populate(next_scans, reserve_jobs=True, suppress_errors=True)
# # tune.MovieOracleTimeCourse.populate(next_scans, reserve_jobs=True, suppress_errors=True)
#
# tune.CaTimes.populate(next_scans, reserve_jobs=True, suppress_errors=True)
# tune.Ori.populate(next_scans, reserve_jobs=True, suppress_errors=True)