from scipy.misc import imresize
import datajoint as dj
from datajoint.jobs import key_hash

from . import experiment, notify
from .exceptions import PipelineException

from warnings import warn
import cv2
import numpy as np
from commons import lab

from pipeline.utils import ts2sec, read_video_hdf5

schema = dj.schema('pipeline_posture', locals())


@schema
class Posture(dj.Imported):
    definition = """
    # posture preview and timestamps

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

    def grab_timestamps_and_frames(self, key, n_sample_frames=16):
        rel = experiment.Session() * experiment.Scan.PostureVideo() * experiment.Scan.BehaviorFile().proj(
            hdf_file='filename')

        info = (rel & key).fetch1()

        avi_path = lab.Paths().get_local_path("{behavior_path}/{filename}".format(**info))
        # replace number by %d for hdf-file reader

        tmp = info['hdf_file'].split('.')
        if not '%d' in tmp[0]:
            info['hdf_file'] = tmp[0][:-1] + '%d.' + tmp[-1]

        hdf_path = lab.Paths().get_local_path("{behavior_path}/{hdf_file}".format(**info))

        print('Reading hdf5 files ...')
        data = read_video_hdf5(hdf_path)
        posture_time, _ = ts2sec(data['posture_ts'][0])

        total_frames = len(posture_time)
        frame_idx = np.floor(np.linspace(0, total_frames - 1, n_sample_frames))


        cap = cv2.VideoCapture(avi_path)
        no_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames != no_frames:
            warn("{total_frames} timestamps, but {no_frames}  movie frames.".format(total_frames=total_frames,
                                                                                    no_frames=no_frames))
            if total_frames > no_frames and total_frames and no_frames:
                total_frames = no_frames
                posture_time = posture_time[:total_frames]
                frame_idx = np.round(np.linspace(0, total_frames - 1, n_sample_frames)).astype(int)
            else:
                raise PipelineException('Can not reconcile frame count', key)
        frames = []
        for frame_pos in frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()

            frames.append(np.asarray(frame, dtype=float)[..., 0])
        frames = np.stack(frames, axis=2)

        return posture_time, frames, total_frames

    def _make_tuples(self, key):
        key['posture_time'], key['preview_frames'], key['total_frames'] = self.grab_timestamps_and_frames(key)

        self.insert1(key)
        del key['posture_time']
        frames = key.pop('preview_frames')
        self.notify(key, frames)

    @notify.ignore_exceptions
    def notify(self, key, frames):
        import imageio

        video_filename = '/tmp/' + key_hash(key) + '.gif'
        frames = frames.transpose([2, 0, 1])
        frames = [imresize(img, 0.25) for img in frames]
        imageio.mimsave(video_filename, frames, duration=0.5)

        msg = 'posture frames for {animal_id}-{session}-{scan_idx}'.format(**key)
        slack_user = notify.SlackUser() & (experiment.Session() & key)
        slack_user.notify(file=video_filename, file_title=msg, channel='#pipeline_quality')

    def get_video_path(self):
        video_info = (experiment.Session() * experiment.Scan.PostureVideo() & self).fetch1()
        return lab.Paths().get_local_path("{behavior_path}/{filename}".format(**video_info))

