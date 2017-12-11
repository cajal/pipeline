#!/usr/local/bin/python
from pipeline import reso, meso, stack
from pipeline import experiment
import time

while True:
    # Scans
    for priority in range(120, -130, -10): # highest to lowest priority
        next_scans = experiment.AutoProcessing() & 'priority > {}'.format(priority)
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

    # Stacks
    stack.StackInfo().populate(stack.CorrectionChannel(), reserve_jobs=True, suppress_errors=True) #TODO: stackAutoProcessing
    stack.Quality().populate(reserve_jobs=True, suppress_errors=True)
    stack.RasterCorrection().populate(reserve_jobs=True, suppress_errors=True)
    stack.MotionCorrection().populate(reserve_jobs=True, suppress_errors=True)
    stack.Stitching().populate(reserve_jobs=True, suppress_errors=True)
    stack.CorrectedStack().populate(reserve_jobs=True, suppress_errors=True)

    time.sleep(60) # wait a minute before trying to process things again
