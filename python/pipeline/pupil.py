from pprint import pprint

from scipy.misc import imresize

import datajoint as dj
from datajoint.jobs import key_hash
from . import experiment, notify
from .exceptions import PipelineException

from warnings import warn
import numpy as np
import json
import os
from commons import lab
import sh

try:
    import pyfnnd
except ImportError:
    warn('Could not load pyfnnd.  Oopsi spike inference will fail. Install from https://github.com/cajal/PyFNND.git')

from .utils.eye_tracking import ROIGrabber, PupilTracker, CVROIGrabber
from pipeline.utils import ts2sec, read_video_hdf5
from . import config

schema = dj.schema('pipeline_eye', locals())

DEFAULT_PARAMETERS = dict(relative_area_threshold=0.002,
                          ratio_threshold=1.5,
                          error_threshold=0.1,
                          min_contour_len=5,
                          margin=0.02,
                          contrast_threshold=5,
                          speed_threshold=0.1,
                          dr_threshold=0.1,
                          gaussian_blur=5,
                          extreme_meso=0)


@schema
class Eye(dj.Imported):
    definition = """
    # eye velocity and timestamps

    -> experiment.Scan
    ---
    total_frames                : int       # total number of frames in movie.
    preview_frames              : longblob  # 16 preview frames 
    eye_time                    : longblob  # timestamps of each frame in seconds, with same t=0 as patch and ball data
    eye_ts=CURRENT_TIMESTAMP    : timestamp # automatic
    """

    @property
    def key_source(self):
        return (experiment.Scan() & experiment.Scan.EyeVideo().proj()) - experiment.ScanIgnored()

    def grab_timestamps_and_frames(self, key, n_sample_frames=16):

        import cv2

        rel = experiment.Session() * experiment.Scan.EyeVideo() * experiment.Scan.BehaviorFile().proj(
            hdf_file='filename')

        info = (rel & key).fetch1()

        avi_path = lab.Paths().get_local_path("{behavior_path}/{filename}".format(**info))
        # replace number by %d for hdf-file reader

        tmp = info['hdf_file'].split('.')
        if not '%d' in tmp[0]:
            info['hdf_file'] = tmp[0][:-1] + '%d.' + tmp[-1]

        hdf_path = lab.Paths().get_local_path("{behavior_path}/{hdf_file}".format(**info))

        data = read_video_hdf5(hdf_path)
        packet_length = data['analogPacketLen']
        dat_time, _ = ts2sec(data['ts'], packet_length)

        if float(data['version']) == 2.:
            cam_key = 'eyecam_ts'
            eye_time, _ = ts2sec(data[cam_key][0])
        else:
            cam_key = 'cam1ts' if info['rig'] == '2P3' else  'cam2ts'
            eye_time, _ = ts2sec(data[cam_key])

        total_frames = len(eye_time)

        frame_idx = np.floor(np.linspace(0, total_frames - 1, n_sample_frames))

        cap = cv2.VideoCapture(avi_path)
        no_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames != no_frames:
            warn("{total_frames} timestamps, but {no_frames}  movie frames.".format(total_frames=total_frames,
                                                                                    no_frames=no_frames))
            if total_frames > no_frames and total_frames and no_frames:
                total_frames = no_frames
                eye_time = eye_time[:total_frames]
                frame_idx = np.round(np.linspace(0, total_frames - 1, n_sample_frames)).astype(int)
            else:
                raise PipelineException('Can not reconcile frame count', key)
        frames = []
        for frame_pos in frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()

            frames.append(np.asarray(frame, dtype=float)[..., 0])
        frames = np.stack(frames, axis=2)

        return eye_time, frames, total_frames

    def _make_tuples(self, key):
        key['eye_time'], key['preview_frames'], key['total_frames'] = self.grab_timestamps_and_frames(key)

        self.insert1(key)
        del key['eye_time']
        frames = key.pop('preview_frames')
        self.notify(key, frames)

    def notify(self, key, frames):
        import imageio
        msg = 'Eye for `{}` has been populated. You can add a tracking task now. '.format(key)
        img_filename = '/tmp/' + key_hash(key) + '.gif'
        frames = frames.transpose([2, 0, 1])
        frames = [imresize(img, 0.25) for img in frames]
        imageio.mimsave(img_filename, frames, duration=0.5)
        (notify.SlackUser() & (experiment.Session() & key)).notify(msg, file=img_filename,
                                                                   file_title='preview frames')

    def get_video_path(self):
        video_info = (experiment.Session() * experiment.Scan.EyeVideo() & self).fetch1()
        return lab.Paths().get_local_path("{behavior_path}/{filename}".format(**video_info))


@schema
class TrackingTask(dj.Manual):
    definition = """
    # ROI and parameters for tracking the eye
    -> Eye
    ---
    eye_roi                     : tinyblob  # manual roi containing eye in full-size movie
    """

    class ManualParameters(dj.Part):
        definition = """
        # manual tracking parameters overwriting the default settings
        -> master
        ---
        tracking_parameters  : varchar(512)  # tracking parameters
        """

    class Ignore(dj.Part):
        definition = """
        # eyes that are too bad to be tracked
        -> master
        ---
        """

    class Mask(dj.Part):
        definition = """
        # mask for tracking
        -> master
        ---
        mask        : longblob 
        """


    @staticmethod
    def _get_modified_parameters():
        new_param = dict(DEFAULT_PARAMETERS)
        for k, v in new_param.items():
            nv = input("{} (default: {}): ".format(k, v))
            new_param[k] = float(nv) if nv else v
        return json.dumps(new_param)

    def enter_roi(self, key, **kwargs):
        key = (Eye() & key).fetch1(dj.key)  # complete key
        frames = (Eye() & key).fetch1('preview_frames')
        try:
            import cv2
            print('Drag window and print q when done')
            rg = CVROIGrabber(frames.mean(axis=2))
            rg.grab()
        except ImportError:
            rg = ROIGrabber(frames.mean(axis=2))

        key['eye_roi'] = rg.roi
        mask = np.asarray(rg.mask, dtype=np.uint8)
        with self.connection.transaction:
            self.insert1(key)
            trackable = input('Is the quality good enough to be tracked? [Y/n]')
            if trackable.lower() == 'n':
                self.insert1(key)
                self.Ignore().insert1(key, ignore_extra_field=True)
            else:
                new_param = dict(DEFAULT_PARAMETERS, **kwargs)
                print('Those are the tracking parameters')
                pprint(new_param)
                new_param = json.dumps(new_param)
                extra_parameters = input('Do you want to change them? [N/y]')
                if extra_parameters.lower() == 'y':
                    new_param = self._get_modified_parameters()
                self.ManualParameters().insert1(dict(key, tracking_parameters=new_param),
                                                ignore_extra_fields=True)
            if np.any(mask == 0):
                print('Inserting mask')
                key['mask'] = mask
                self.Mask().insert1(key, ignore_extra_fields=True)


@schema
class TrackedVideo(dj.Computed):
    definition = """
    -> Eye
    -> TrackingTask
    ---
    tracking_parameters              : longblob  # tracking parameters
    tracking_ts=CURRENT_TIMESTAMP    : timestamp  # automatic
    """

    class Frame(dj.Part):
        definition = """
        -> TrackedVideo
        frame_id                 : int           # frame id with matlab based 1 indexing
        ---
        rotated_rect=NULL        : tinyblob      # rotated rect (center, sidelength, angle) containing the ellipse
        contour=NULL             : longblob      # eye contour relative to ROI
        center=NULL              : tinyblob      # center of the ellipse in (x, y) of image
        major_r=NULL             : float         # major radius of the ellipse
        frame_intensity=NULL     : float         # std of the frame
        """

    key_source = Eye() * TrackingTask() - TrackingTask.Ignore()

    def _make_tuples(self, key):
        print("Populating", key)
        param = DEFAULT_PARAMETERS
        if TrackingTask.ManualParameters() & key:
            param = json.loads((TrackingTask.ManualParameters() & key).fetch1('tracking_parameters'))
            print('Using manual set parameters', param, flush=True)

        roi = (TrackingTask() & key).fetch1('eye_roi')

        avi_path = (Eye() & key).get_video_path()
        print(avi_path)

        if TrackingTask.Mask() & key:
            mask = (TrackingTask.Mask() & key).fetch1('mask')
        else:
            mask = None

        tr = PupilTracker(param, mask=mask)
        traces = tr.track(avi_path, roi - 1, display=config['display.tracking'])  # -1 because of matlab indices

        key['tracking_parameters'] = json.dumps(param)
        self.insert1(key)
        fr = self.Frame()
        for trace in traces:
            trace.update(key)
            fr.insert1(trace, ignore_extra_fields=True)

        (notify.SlackUser() & (experiment.Session() & key)).notify(
            'Pupil tracking for {} has been populated'.format(str(key)))

    def plot_traces(self, outdir='./', show=False):
        """
        Plot existing traces to output directory.

        :param outdir: destination of plots
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        plt.switch_backend('GTK3Agg')

        for key in self.fetch.keys():
            print('Processing', key)
            with sns.axes_style('ticks'):
                fig, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

            r, center, contrast = (TrackedVideo.Frame() & key).fetch('major_r', 'center',
                                                                     'frame_intensity', order_by='frame_id')
            ax[0].plot(r)
            ax[0].set_title('Major Radius')
            c = np.vstack([cc if cc is not None else np.NaN * np.ones(2) for cc in center])

            ax[1].plot(c[:, 0], label='x')
            ax[1].plot(c[:, 1], label='y')
            ax[1].set_title('Pupil Center Coordinates')
            ax[1].legend()

            ax[2].plot(contrast)
            ax[2].set_title('Contrast (frame std)')
            ax[2].set_xlabel('Frames')
            try:
                sh.mkdir('-p', os.path.expanduser(outdir) + '/{animal_id}/'.format(**key))
            except:
                pass

            fig.suptitle(
                'animal id {animal_id} session {session} scan_idx {scan_idx} eye quality {eye_quality}'.format(**key))
            fig.tight_layout()
            sns.despine(fig)
            fig.savefig(outdir + '/{animal_id}/AI{animal_id}SE{session}SI{scan_idx}EQ{eye_quality}.png'.format(**key))
            if show:
                plt.show()
            else:
                plt.close(fig)

    def show_video(self, from_frame, to_frame, framerate=1000):
        """
        Shows the video from from_frame to to_frame (1-based) and the corrsponding tracking results.
        Needs opencv installation.

        :param from_frame: start frame (1 based)
        :param to_frame:  end frame (1 based)
        """
        if not len(self) == 1:
            raise PipelineException("Restrict EyeTracking to one video only.")
        import cv2
        video_info = (experiment.Session() * experiment.Scan.EyeVideo() & self).fetch1()
        videofile = "{path_prefix}/{behavior_path}/{filename}".format(path_prefix=config['path.mounts'], **video_info)
        eye_roi = (Eye() & self).fetch1('eye_roi') - 1

        contours, ellipses = ((TrackedVideo.Frame() & self) \
                              & 'frame_id between {0} and {1}'.format(from_frame, to_frame)
                              ).fetch('contour', 'rotated_rect', order_by='frame_id')
        cap = cv2.VideoCapture(videofile)
        no_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        font = cv2.FONT_HERSHEY_SIMPLEX
        if not from_frame < no_frames:
            raise PipelineException('Starting frame exceeds number of frames')

        cap.set(cv2.CAP_PROP_POS_FRAMES, from_frame - 1)
        fr_count = from_frame - 1

        elem_count = 0
        while cap.isOpened():
            fr_count += 1
            ret, frame = cap.read()
            if fr_count < from_frame:
                continue

            if fr_count >= to_frame or fr_count >= no_frames:
                print("Reached end of videofile ", videofile)
                break
            contour = contours[elem_count]
            ellipse = ellipses[elem_count]
            elem_count += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.putText(gray, str(fr_count), (10, 30), font, 1, (127, 127, 127), 2)

            if contour is not None:
                ellipse = (tuple(eye_roi[::-1, 0] + ellipse[:2]), tuple(ellipse[2:4]), ellipse[4])
                cv2.drawContours(gray, [contour.astype(np.int32)], 0, (255, 0, 0), 1, offset=tuple(eye_roi[::-1, 0]))
                cv2.ellipse(gray, ellipse, (0, 0, 255), 2)
            cv2.imshow('frame', gray)

            if (cv2.waitKey(int(1000 / framerate)) & 0xFF == ord('q')):
                break

        cap.release()
        cv2.destroyAllWindows()
