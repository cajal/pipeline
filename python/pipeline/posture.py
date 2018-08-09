from scipy.misc import imresize
import datajoint as dj
from datajoint.jobs import key_hash
import cv2
import numpy as np
from commons import lab
import os

from .utils import h5
from . import experiment, notify
from .exceptions import PipelineException


schema = dj.schema('pipeline_posture', locals())


@schema
class Posture(dj.Imported):
    definition = """ # posture preview and timestamps

    -> experiment.Scan
    ---
    total_frames                    : int       # total number of frames in movie.
    preview_frames                  : longblob  # 16 preview frames
    posture_time                    : longblob  # timestamps of each frame in seconds, with same t=0 as patch and ball data
    posture_ts=CURRENT_TIMESTAMP    : timestamp # automatic
    """
    @property
    def key_source(self):
        return experiment.Scan() & experiment.Scan.PostureVideo().proj()

    def _make_tuples(self, key):
        # Get behavior filename
        behavior_path = (experiment.Session() & key).fetch1('behavior_path')
        local_path = lab.Paths().get_local_path(behavior_path)
        filename = (experiment.Scan.BehaviorFile() & key).fetch1('filename')
        full_filename = os.path.join(local_path, filename)

        # Read file
        data = h5.read_behavior_file(full_filename)

        # Get counter timestamps and convert to seconds
        timestamps_in_secs = h5.ts2sec(data['posture_ts'][0])
        ts = h5.ts2sec(data['ts'], is_packeted=True)
        # edge case when ts and eye ts start in different sides of the master clock max value 2 **32
        if abs(ts[0] - timestamps_in_secs[0]) > 2 ** 31:
            timestamps_in_secs += (2 ** 32 if ts[0] > timestamps_in_secs[0] else -2 ** 32)

        # Fill with NaNs for out-of-range data or mistimed packets (NaNs in ts)
        timestamps_in_secs[timestamps_in_secs < ts[0]] = float('nan')
        timestamps_in_secs[timestamps_in_secs > ts[-1]] = float('nan')
        nan_limits = np.where(np.diff([0, *np.isnan(ts), 0]))[0]
        for start, stop in zip(nan_limits[::2], nan_limits[1::2]):
            lower_ts = float('-inf') if start == 0 else ts[start - 1]
            upper_ts = float('inf') if stop == len(ts) else ts[stop]
            timestamps_in_secs[np.logical_and(timestamps_in_secs > lower_ts,
                                              timestamps_in_secs < upper_ts)] = float('nan')

        # Read video
        filename = (experiment.Scan.PostureVideo() & key).fetch1('filename')
        full_filename = os.path.join(local_path, filename)
        video = cv2.VideoCapture(full_filename)

        # Fix inconsistent num_video_frames vs num_timestamps
        num_video_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        num_timestamps = len(timestamps_in_secs)
        if num_timestamps != num_video_frames:
            if abs(num_timestamps - num_video_frames) > 1:
                msg = ('Number of movie frames and timestamps differ: {} frames vs {} '
                       'timestamps'). format(num_video_frames, num_timestamps)
                raise PipelineException(msg)
            elif num_timestamps > num_video_frames: # cut timestamps to match video frames
                timestamps_in_secs = timestamps_in_secs[:-1]
            else: # fill with NaNs
                timestamps_in_secs[-1] = float('nan')

        # Get 16 sample frames
        frames = []
        for frame_idx in np.round(np.linspace(0, num_video_frames - 1, 16)).astype(int):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            _, frame = video.read()
            frames.append(np.asarray(frame, dtype=float)[..., 0])
        frames = np.stack(frames, axis=-1)

        # Insert
        self.insert1({**key, 'posture_time': timestamps_in_secs,
                      'total_frames': len(timestamps_in_secs), 'preview_frames': frames})
        self.notify(key, frames)

    @notify.ignore_exceptions
    def notify(self, key, frames):
        import imageio

        video_filename = '/tmp/' + key_hash(key) + '.gif'
        frames = [imresize(img, 0.25) for img in frames.transpose([2, 0, 1])]
        imageio.mimsave(video_filename, frames, duration=0.5)

        msg = 'posture frames for {animal_id}-{session}-{scan_idx}'.format(**key)
        slack_user = notify.SlackUser() & (experiment.Session() & key)
        slack_user.notify(file=video_filename, file_title=msg, channel='#pipeline_quality')

    def get_video_path(self):
        video_info = (experiment.Session() * experiment.Scan.PostureVideo() & self).fetch1()
        video_path = lab.Paths().get_local_path("{behavior_path}/{filename}".format(**video_info))
        return video_path

