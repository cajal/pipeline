#!/usr/local/bin/python
from pipeline import reso, meso, fuse, stack
from pipeline import experiment
import time
import warnings

while True:
    # Scans
    for priority in range(120, -130, -10): # highest to lowest priority
        next_scans = experiment.AutoProcessing() & 'priority > {}'.format(priority)

        # reso/meso
        for pipe in [reso, meso]:
            pipe.ScanInfo().populate(next_scans, reserve_jobs=True, suppress_errors=True)
            pipe.Quality().populate(next_scans, reserve_jobs=True, suppress_errors=True)
            pipe.RasterCorrection().populate(next_scans, reserve_jobs=True, suppress_errors=True)
            pipe.MotionCorrection().populate(next_scans, reserve_jobs=True, suppress_errors=True)
            pipe.SummaryImages().populate(next_scans, reserve_jobs=True, suppress_errors=True)
            pipe.Segmentation().populate(next_scans, reserve_jobs=True, suppress_errors=True)
            pipe.MaskClassification().populate(next_scans, {'classification_method': 2}, reserve_jobs=True, suppress_errors=True)
            pipe.ScanSet().populate(next_scans, reserve_jobs=True, suppress_errors=True)
            pipe.Activity().populate(next_scans, {'spike_method': 5}, reserve_jobs=True, suppress_errors=True)
            full_scans = (pipe.ScanInfo().proj() & pipe.Activity()) - (pipe.ScanInfo.Field() - pipe.Activity())
            pipe.ScanDone().populate(next_scans & full_scans, reserve_jobs=True, suppress_errors=True)

        # fuse
        fuse.MotionCorrection().populate(next_scans, reserve_jobs=True, suppress_errors=True)
        fuse.ScanSet().populate(next_scans, reserve_jobs=True, suppress_errors=True)
        fuse.Activity().populate(next_scans, reserve_jobs=True, suppress_errors=True)
        fuse.ScanDone().populate(next_scans, reserve_jobs=True, suppress_errors=True)

        # tune (some of these are memory intensive)
        next_scans = next_scans & (experiment.Scan() & 'scan_ts > "2017-12-00 00:00:00"')
        try:
            from stimline import tune
        except ImportError:
            warnings.warn('Warning: Skipping pixelwise maps. Install stimulus (cajal/stimuli)'
                          ' and stimline (cajal/stimulus-pipeline).')
        else:
            #stimulus.Sync needs to be ran from Matlab
            tune.STA().populate(next_scans, reserve_jobs=True, suppress_errors=True)
            tune.STAQual().populate(next_scans, reserve_jobs=True, suppress_errors=True)
            tune.TimedSTA().populate(next_scans, reserve_jobs=True, suppress_errors=True)
            tune.TimedSTAQual().populate(next_scans, reserve_jobs=True, suppress_errors=True)

            tune.CaMovie().populate(next_scans, reserve_jobs=True, suppress_errors=True)
            tune.Drift().populate(next_scans, reserve_jobs=True, suppress_errors=True)
            tune.OriDesign().populate(next_scans, reserve_jobs=True, suppress_errors=True)
            tune.OriMap().populate(next_scans, reserve_jobs=True, suppress_errors=True)
            tune.Cos2Map().populate(next_scans, reserve_jobs=True, suppress_errors=True)
            tune.OriMapQuality().populate(next_scans, reserve_jobs=True, suppress_errors=True)

            tune.OracleMap().populate(next_scans, reserve_jobs=True, suppress_errors=True)

    # Stacks
    stack.StackInfo().populate(stack.CorrectionChannel(), reserve_jobs=True, suppress_errors=True) #TODO: stackAutoProcessing
    stack.Quality().populate(reserve_jobs=True, suppress_errors=True)
    stack.RasterCorrection().populate(reserve_jobs=True, suppress_errors=True)
    stack.MotionCorrection().populate(reserve_jobs=True, suppress_errors=True)
    stack.Stitching().populate(reserve_jobs=True, suppress_errors=True)
    stack.CorrectedStack().populate(reserve_jobs=True, suppress_errors=True)

    time.sleep(60) # wait a minute before trying to process things again
