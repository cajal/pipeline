#!/usr/local/bin/python3
import time 

from pipeline import stack

while True: 
    stack.Segmentation.populate(reserve_jobs=True, suppress_errors=True) 
    time.sleep(3600) # wait an hour before checking again
