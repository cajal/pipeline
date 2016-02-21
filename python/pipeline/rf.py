import warnings
import datajoint as dj
from pipeline.lib import ROIGrabber

schema = dj.schema('pipeline_rf', locals())
from . import lib

import matplotlib.pyplot as plt
import h5py
import seaborn as sns
import numpy as np

try:
    import cv2
except:
    warnings.warn("OpenCV is not installed. You won't be able to populate rf.Eye")

@schema
class Eye(dj.Imported):
    definition = ...

    @property
    def populated_from(self):
        return Scan()

    def _make_tuples(self, key):
        p, f = (Session() & key).fetch1['hd5_path','file_base']
        n = (Scan() & key).fetch1['file_num']
        avi_path = r"{p}/{f}{n}eyetracking.avi".format(f=f, p=p, n=n)
        hdf_path = r"{p}/{f}{n}%d.h5".format(f=f, p=p, n=n)
        data = lib.read_video_hdf5(hdf_path)

        packet_length = data['analogPacketLen']
        dat_time,_,_ = lib.ts2sec(data['ts'], packet_length)
        eye_time,_,_ = lib.ts2sec(data['cam2ts'], packet_length)
        total_frames = len(eye_time)
        n_sample_frames = 10
        frame_idx = np.round(np.linspace(1,total_frames,n_sample_frames))

        cap = cv2.VideoCapture(avi_path)
        no_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames != no_frames:
            warnings.warn("{total_frames} timestamps, but {no_frames}  movie frames.".format(total_frames=total_frames, no_frames=no_frames))
            if total_frames > no_frames and total_frames and no_frames:
                total_frames = no_frames
                eye_time = eye_time[:total_frames]
                frame_idx = np.round(np.linspace(0,total_frames-1,n_sample_frames)).astype(int)
            else:
                raise Exception('Can not reconcile frame count')
        frames = []
        for frame_pos in frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            frames.append(np.asarray(frame)[...,0])
        frames = np.stack(frames, axis=2)

        rg = ROIGrabber(frames.mean(axis=2))

        roi = rg.roi
        print(roi)
        key.eye_time = eye_time
        key.eye_roi = roi
        #self.insert1(key)
