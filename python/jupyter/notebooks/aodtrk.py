import datajoint as dj
import pandas as pd
from pipeline import aodpre
#from aodpre import *
import warnings
from IPython import embed
import glob
import numpy as np
import dateutil.parser
from pipeline import utils

try:
    from pupil_tracking import PupilTracker
except ImportError:
    warnings.warn("Failed to import pupil_tacking library. You won't be able to populate trk.EyeFrame")

schema = dj.schema('pipeline_aod_pupiltracking', locals())

@schema
class TrackInfo(dj.Imported):
    definition = """
    # machine independent path of eye videos
    
    ->aodpre.Scan
    ---
    base_video_path: varchar(100) # base path of the video
    """

    def _make_tuples(self, key):
        print("key = ", key)
        # embed()
        path = (aodpre.Scan() & key).fetch1['hdf5_file']
        words = path.split('\\')
        if len(words) == 1:
            words = words[0].split('/')
        i = words.index('Mouse')
        ymd = words[i+3].split('_')[0]
        hms = words[i+3].split('_')[1].replace("-", ":")
        time_hdf5 = dateutil.parser.parse("{ymd} {hms}".format(ymd=ymd,hms=hms))


        # time_str = words[i+3].split('_')[1].split('-')
        # time_hdf5 = int(time_str[0])*10000 + int(time_str[1])*100 + int(time_str[2])
        folders = glob.glob(r"/m/Mouse/{f1}/20*".format(f1=words[i+1]))
        time_coll = []
        time_diff = []
        for name in folders:
            ymd = name.split('/')[4].split('_')[0]
            hms = name.split('/')[4].split('_')[1].replace("-", ":")
            # t = name.split('/')[-1].split('_')[1].split('-')
            # time = int(t[0])*10000 + int(t[1])*100 + int(t[2])
            time = dateutil.parser.parse("{ymd} {hms}".format(ymd=ymd,hms=hms))
            time_coll.append(time)
            diff = abs((time_hdf5 - time).total_seconds())
            time_diff.append(diff)

        time_diff = np.asarray(time_diff)
        fo = folders[np.argmin(abs(time_diff))]
        avi_path = glob.glob(r"{fo}/*.avi".format(fo=fo))
        assert len(avi_path) == 1, "Found 0 or more than 1 videos: {videos}".format(videos=str(avi_path))
        key['base_video_path'] = avi_path[0]
        self.insert1(key)

    def get_frames(self, key):
        path = (aodpre.Scan() & key).fetch1['hdf5_file']
        video_file = (self & key).fetch1['base_video_path']
        # embed()
        import cv2
        cap = cv2.VideoCapture(video_file)
        fr_count = 0
        while cap.isOpened():
            fr_count += 1
            ret, frame = cap.read()
            if fr_count == 1000:
                return frame

        # data = utils.read_video_hdf5(hdf_path)


@schema
class Roi(dj.Manual):
    definition = """
    # table that stores the correct ROI of the Eye in the video
    ->TrackInfo
    ---
    x_roi_min                     : int                         # x coordinate of roi
    y_roi_min                     : int                         # y coordinate of roi
    x_roi_max                     : int                         # x coordinate of roi
    y_roi_max                     : int                         # y coordinate of roi
    """

# embed()

@schema
class ParamEyeFrame(dj.Lookup):
    definition = """
    # table that stores the paths for the params for pupil_tracker
    param_id                      : int            # id for param collection
    ---
    weight_thres_low = Null                 : int        # parameter for tracking
    weight_thres_high = Null                 : int        # parameter for tracking
    thres_per_high = Null                 : int        # parameter for tracking
    thres_per_low = Null                 : int        # parameter for tracking
    """

    contents = [
        {'param_id': 0, 'weight_thres_low': 0.5,  'weight_thres_high': 0.5,  'thres_per_high': 99, 'thres_per_low': 1},
        {'param_id': 1, 'weight_thres_low': 0.75, 'weight_thres_high': 0.25, 'thres_per_high': 97, 'thres_per_low': 3}
    ]


@schema
class EyeFrame(dj.Computed):
    definition = """
    # eye tracking info for each frame of a movie
    -> Roi
    -> ParamEyeFrame
    frame                       : int                           # frame number in movie
    ---
    eye_frame_ts=CURRENT_TIMESTAMP    : timestamp               # automatic
    """

    @property
    def populated_from(self):
        return Roi()

    def _make_tuples(self, key):
        pass



