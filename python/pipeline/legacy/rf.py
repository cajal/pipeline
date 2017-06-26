"""
This is a legacy schema that is no longer used in the unified pipeline.
"""

import warnings
import datajoint as dj
import pipeline.utils.h5
from pipeline.utils import ROIGrabber

schema = dj.schema('pipeline_rf', locals())
from .. import utils
import numpy as np
import os

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

    def unpopulated(self, path_prefix=''):
        """
        Returns all keys from Scan()*Session() that are not in Eye but have a video.


        :param path_prefix: prefix to the path to find the video (usually '/mnt/', but empty by default)
        """

        return (Scan() * Session()).project() &  [k for k in (Scan() * Session() - self).project('hd5_path','file_base','file_num').fetch.as_dict() if
                    os.path.exists("{hd5_path}/{file_base}{file_num}eyetracking.avi".format(**k))]

    def new_eye(self, key):
        p, f = (Session() & key).fetch1['hd5_path', 'file_base']
        n = (Scan() & key).fetch1['file_num']
        avi_path = r"{p}/{f}{n}eyetracking.avi".format(f=f, p=p, n=n)
        hdf_path = r"{p}/{f}{n}%d.h5".format(f=f, p=p, n=n)
        data = pipeline.utils.h5.read_video_hdf5(hdf_path)

        packet_length = data['analogPacketLen']
        dat_time, _, _ = pipeline.utils.h5.ts2sec(data['ts'], packet_length)
        eye_time, _, _ = pipeline.utils.h5.ts2sec(data['cam2ts'], packet_length)
        total_frames = len(eye_time)
        n_sample_frames = 10
        frame_idx = np.round(np.linspace(1, total_frames, n_sample_frames))

        cap = cv2.VideoCapture(avi_path)
        no_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames != no_frames:
            warnings.warn("{total_frames} timestamps, but {no_frames}  movie frames.".format(total_frames=total_frames,
                                                                                             no_frames=no_frames))
            if total_frames > no_frames and total_frames and no_frames:
                total_frames = no_frames
                eye_time = eye_time[:total_frames]
                frame_idx = np.round(np.linspace(0, total_frames - 1, n_sample_frames)).astype(int)
            else:
                raise Exception('Can not reconcile frame count')
        frames = []
        for frame_pos in frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            frames.append(np.asarray(frame)[..., 0])
        frames = np.stack(frames, axis=2)
        return eye_time, frames

    def _make_tuples(self, key):
        pass

schema.spawn_missing_classes()
