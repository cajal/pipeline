from .exceptions import PipelineException
from . import experiment, notify
from .utils import h5
from . import config
from .utils.eye_tracking import PupilTracker, ManualTracker
from .utils import eye_tracking
from .utils.decorators import gitlog

import datajoint as dj
from datajoint.autopopulate import AutoPopulate
from datajoint.jobs import key_hash

from commons import lab
import json
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

from scipy.misc import imresize
from itertools import count

# Disable DLC GUI first, then import deeplabcut
import os
os.environ["DLClight"] = "True"
import deeplabcut as dlc
from deeplabcut.utils import auxiliaryfunctions
from .utils import DLC_tools


schema = dj.schema('pipeline_eye', locals())

gputouse = 0

DEFAULT_PARAMETERS = {'relative_area_threshold': 0.002,
                      'ratio_threshold': 1.5,
                      'error_threshold': 0.1,
                      'min_contour_len': 5,
                      'margin': 0.02,
                      'contrast_threshold': 5,
                      'speed_threshold': 0.1,
                      'dr_threshold': 0.1,
                      'gaussian_blur': 5,
                      'extreme_meso': 0,
                      'running_avg': 0.4,
                      'exponent': 9
                      }


@schema
class Eye(dj.Imported):
    definition = """  # eye movie timestamps synchronized to behavior clock

    -> experiment.Scan
    ---
    eye_time                    : longblob      # times of each movie frame in behavior clock
    total_frames                : int           # number of frames in movie.
    preview_frames              : longblob      # 16 preview frames
    eye_ts=CURRENT_TIMESTAMP    : timestamp
    """

    @property
    def key_source(self):
        return experiment.Scan() & experiment.Scan.EyeVideo().proj()

    def _make_tuples(self, key):
        # Get behavior filename
        behavior_path = (experiment.Session() & key).fetch1('behavior_path')
        local_path = lab.Paths().get_local_path(behavior_path)
        filename = (experiment.Scan.BehaviorFile() & key).fetch1('filename')
        full_filename = os.path.join(local_path, filename)

        # Read file
        data = h5.read_behavior_file(full_filename)

        # Get counter timestamps and convert to seconds
        if data['version'] == '1.0':  # older h5 format
            rig = (experiment.Session() & key).fetch('rig')
            timestamps_in_secs = h5.ts2sec(
                data['cam1_ts' if rig == '2P3' else 'cam2_ts'])
        else:
            timestamps_in_secs = h5.ts2sec(data['eyecam_ts'][0])
        ts = h5.ts2sec(data['ts'], is_packeted=True)
        # edge case when ts and eye ts start in different sides of the master clock max value 2 **32
        if abs(ts[0] - timestamps_in_secs[0]) > 2 ** 31:
            timestamps_in_secs += (2 ** 32 if ts[0]
                                   > timestamps_in_secs[0] else -2 ** 32)

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
        filename = (experiment.Scan.EyeVideo() & key).fetch1('filename')
        full_filename = os.path.join(local_path, filename)
        # note: prints many 'Unexpected list ...'
        video = cv2.VideoCapture(full_filename)

        # Fix inconsistent num_video_frames vs num_timestamps
        num_video_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        num_timestamps = len(timestamps_in_secs)
        if num_timestamps != num_video_frames:
            if abs(num_timestamps - num_video_frames) > 1:
                msg = ('Number of movie frames and timestamps differ: {} frames vs {} '
                       'timestamps'). format(num_video_frames, num_timestamps)
                raise PipelineException(msg)
            elif num_timestamps > num_video_frames:  # cut timestamps to match video frames
                timestamps_in_secs = timestamps_in_secs[:-1]
            else: # fill with NaNs
                timestamps_in_secs = np.array(
                    [*timestamps_in_secs, float('nan')])

        # Get 16 sample frames
        frames = []
        for frame_idx in np.round(np.linspace(0, num_video_frames - 1, 16)).astype(int):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            _, frame = video.read()
            frames.append(np.asarray(frame, dtype=float)[..., 0])
        frames = np.stack(frames, axis=-1)

        # Insert
        self.insert1({**key, 'eye_time': timestamps_in_secs,
                      'total_frames': len(timestamps_in_secs), 'preview_frames': frames})
        self.notify(key, frames)

    @notify.ignore_exceptions
    def notify(self, key, frames):
        import imageio

        video_filename = '/tmp/' + key_hash(key) + '.gif'
        frames = [imresize(img, 0.25) for img in frames.transpose([2, 0, 1])]
        imageio.mimsave(video_filename, frames, duration=0.5)

        msg = 'eye frames for {animal_id}-{session}-{scan_idx}'.format(**key)
        slack_user = notify.SlackUser() & (experiment.Session() & key)
        slack_user.notify(file=video_filename, file_title=msg,
                          channel='#pipeline_quality')

    def get_video_path(self):
        video_info = (experiment.Session() *
                      experiment.Scan.EyeVideo() & self).fetch1()
        video_path = lab.Paths().get_local_path(
            "{behavior_path}/{filename}".format(**video_info))
        return video_path


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

        print('Drag window and print q when done')
        rg = eye_tracking.CVROIGrabber(frames.mean(axis=2))
        rg.grab()

        key['eye_roi'] = rg.roi
        mask = np.asarray(rg.mask, dtype=np.uint8)
        with self.connection.transaction:
            self.insert1(key)
            trackable = input(
                'Is the quality good enough to be tracked? [Y/n]')
            if trackable.lower() == 'n':
                self.insert1(key)
                self.Ignore().insert1(key, ignore_extra_field=True)
            else:
                new_param = dict(DEFAULT_PARAMETERS, **kwargs)
                print('Those are the tracking parameters')
                print(new_param)
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
    tracking_parameters              : longblob   # tracking parameters
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
            param = json.loads(
                (TrackingTask.ManualParameters() & key).fetch1('tracking_parameters'))
            print('Using manual set parameters', param, flush=True)

        roi = (TrackingTask() & key).fetch1('eye_roi')

        avi_path = (Eye() & key).get_video_path()
        print(avi_path)

        if TrackingTask.Mask() & key:
            mask = (TrackingTask.Mask() & key).fetch1('mask')
        else:
            mask = None

        tr = PupilTracker(param, mask=mask)
        # -1 because of matlab indices
        traces = tr.track(avi_path, roi - 1,
                          display=config['display.tracking'])

        key['tracking_parameters'] = json.dumps(param)
        self.insert1(key)
        fr = self.Frame()
        for trace in traces:
            trace.update(key)
            fr.insert1(trace, ignore_extra_fields=True)

        self.notify(key)

    @notify.ignore_exceptions
    def notify(self, key):
        msg = 'Pupil tracking for {} has been populated'.format(key)
        (notify.SlackUser() & (experiment.Session() & key)).notify(msg)

    def plot_traces(self, outdir='./', show=False):
        """
        Plot existing traces to output directory.

        :param outdir: destination of plots
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        plt.switch_backend('GTK3Agg')

        for key in self.fetch('KEY'):
            print('Processing', key)
            with sns.axes_style('ticks'):
                fig, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

            r, center, contrast = (TrackedVideo.Frame() & key).fetch('major_r', 'center',
                                                                     'frame_intensity', order_by='frame_id')
            ax[0].plot(r)
            ax[0].set_title('Major Radius')
            c = np.vstack([cc if cc is not None else np.NaN *
                           np.ones(2) for cc in center])

            ax[1].plot(c[:, 0], label='x')
            ax[1].plot(c[:, 1], label='y')
            ax[1].set_title('Pupil Center Coordinates')
            ax[1].legend()

            ax[2].plot(contrast)
            ax[2].set_title('Contrast (frame std)')
            ax[2].set_xlabel('Frames')
            try:
                os.mkdirs(os.path.expanduser(outdir) +
                          '/{animal_id}/'.format(**key), exist_ok=True)
            except:
                pass

            fig.suptitle(
                'animal id {animal_id} session {session} scan_idx {scan_idx} eye quality {eye_quality}'.format(**key))
            fig.tight_layout()
            sns.despine(fig)
            fig.savefig(
                outdir + '/{animal_id}/AI{animal_id}SE{session}SI{scan_idx}EQ{eye_quality}.png'.format(**key))
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
        video_info = (experiment.Session() *
                      experiment.Scan.EyeVideo() & self).fetch1()
        videofile = "{path_prefix}/{behavior_path}/{filename}".format(
            path_prefix=config['path.mounts'], **video_info)
        eye_roi = (Eye() & self).fetch1('eye_roi') - 1

        contours, ellipses = ((TrackedVideo.Frame() & self)
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

            cv2.putText(gray, str(fr_count), (10, 30),
                        font, 1, (127, 127, 127), 2)

            if contour is not None:
                ellipse = (
                    tuple(eye_roi[::-1, 0] + ellipse[:2]), tuple(ellipse[2:4]), ellipse[4])
                cv2.drawContours(gray, [contour.astype(
                    np.int32)], 0, (255, 0, 0), 1, offset=tuple(eye_roi[::-1, 0]))
                cv2.ellipse(gray, ellipse, (0, 0, 255), 2)
            cv2.imshow('frame', gray)

            if (cv2.waitKey(int(1000 / framerate)) & 0xFF == ord('q')):
                break

        cap.release()
        cv2.destroyAllWindows()


@schema
@gitlog
class ManuallyTrackedContours(dj.Manual, AutoPopulate):
    definition = """
    -> Eye
    ---
    tracking_ts=CURRENT_TIMESTAMP    : timestamp  # automatic
    min_lambda=null                  : float      # minimum mixing weight for current frame in running average computation (1 means no running avg was used)
    """

    class Frame(dj.Part):
        definition = """
        -> master
        frame_id                 : int           # frame id with matlab based 1 indexing
        ---
        contour=NULL             : longblob      # eye contour relative to ROI
        """

    class Parameter(dj.Part):
        definition = """
        -> master.Frame
        ---
        roi=NULL                : longblob  # roi of eye
        gauss_blur=NULL         : float     # bluring of ROI
        exponent=NULL           : tinyint   # exponent for contrast enhancement
        dilation_iter=NULL      : tinyint   # number of dilation and erosion operations
        min_contour_len=NULL    : tinyint   # minimal contour length
        running_avg_mix=NULL    : float     # weight a in a * current_frame + (1-a) * running_avg
        """

    def make(self, key, backup_file=None):
        print("Populating", key)

        if backup_file is None:
            avi_path = (Eye() & key).get_video_path()
            tracker = ManualTracker(avi_path)
            tracker.backup_file = '/tmp/tracker_state{animal_id}-{session}-{scan_idx}.pkl'.format(
                **key)
        else:
            tracker = ManualTracker.from_backup(backup_file)

        try:
            tracker.run()
        except:
            tracker.backup()
            raise

        logtrace = tracker.mixing_constant.logtrace.astype(float)
        self.insert1(dict(key, min_lambda=logtrace[logtrace > 0].min()))
        self.log_git(key)
        frame = self.Frame()
        parameters = self.Parameter()
        for frame_id, ok, contour, params in tqdm(zip(count(), tracker.contours_detected, tracker.contours,
                                                      tracker.parameter_iter()),
                                                  total=len(tracker.contours)):
            assert frame_id == params['frame_id']
            if ok:
                frame.insert1(dict(key, frame_id=frame_id, contour=contour))
            else:
                frame.insert1(dict(key, frame_id=frame_id))
            parameters.insert1(dict(key, **params), ignore_extra_fields=True)

    def warm_start(self, key, backup_file):
        assert not key in self, '{} should not be in the table already!'
        with self.connection.transaction:
            self.make(key, backup_file)

    def update(self, key):
        print("Populating", key)

        avi_path = (Eye() & key).get_video_path()

        tracker = ManualTracker(avi_path)
        contours = (self.Frame() & key).fetch('contour', order_by='frame_id')
        tracker.contours = np.array(contours)
        tracker.contours_detected = np.array([e is not None for e in contours])
        tracker.backup_file = '/tmp/tracker_update_state{animal_id}-{session}-{scan_idx}.pkl'.format(
            **key)

        try:
            tracker.run()
        except Exception as e:
            print(str(e))
            answer = input(
                'Tracker crashed. Do you want to save the content anyway [y/n]?').lower()
            while answer not in ['y', 'n']:
                answer = input(
                    'Tracker crashed. Do you want to save the content anyway [y/n]?').lower()
            if answer == 'n':
                raise
        if input('Do you want to delete and replace the existing entries? Type "YES" for acknowledgement.') == "YES":
            with dj.config(safemode=False):
                with self.connection.transaction:
                    (self & key).delete()

                    logtrace = tracker.mixing_constant.logtrace.astype(float)
                    self.insert1(
                        dict(key, min_lambda=logtrace[logtrace > 0].min()))
                    self.log_key(key)

                    frame = self.Frame()
                    parameters = self.Parameter()
                    for frame_id, ok, contour, params in tqdm(zip(count(), tracker.contours_detected, tracker.contours,
                                                                  tracker.parameter_iter()),
                                                              total=len(tracker.contours)):
                        assert frame_id == params['frame_id']
                        if ok:
                            frame.insert1(
                                dict(key, frame_id=frame_id, contour=contour))
                        else:
                            frame.insert1(dict(key, frame_id=frame_id))
                        parameters.insert1(
                            dict(key, **params), ignore_extra_fields=True)


@schema
class FittedContour(dj.Computed):
    definition = """
    -> ManuallyTrackedContours
    ---
    tracking_ts=CURRENT_TIMESTAMP    : timestamp  # automatic
    """

    class Ellipse(dj.Part):
        definition = """
        -> master
        frame_id                 : int           # frame id with matlab based 1 indexing
        ---
        center=NULL              : tinyblob      # center of the ellipse in (x, y) of image
        major_r=NULL             : float         # major radius of the ellipse
        """

    def display_frame_number(self, img, frame_number, n_frames):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = .7
        cv2.putText(img, "[{fr_count:05d}/{frames:05d}]".format(
            fr_count=frame_number, frames=n_frames),
            (10, 30), font, fs, (255, 144, 30), 2)

    def make(self, key):
        print("Populating", key)

        avi_path = (Eye() & key).get_video_path()

        contours = (ManuallyTrackedContours.Frame() & key).fetch(
            order_by='frame_id ASC', as_dict=True)
        self._cap = cap = cv2.VideoCapture(avi_path)

        frame_number = 0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        while cap.isOpened():
            if frame_number >= n_frames - 1:
                print("Reached end of videofile ", avi_path)
                break

            ret, frame = self._cap.read()
            ckey = contours[frame_number]
            if ret and frame is not None and ckey['contour'] is not None:
                if ckey['contour'] is not None and len(ckey['contour']) >= 5:
                    contour = ckey['contour']
                    center = contour.mean(axis=0)
                    cv2.drawContours(frame, [contour], -1, (0, 255, 0), 1)
                    cv2.circle(frame, tuple(
                        center.squeeze().astype(int)), 4, (0, 165, 255), -1)
                    ellipse = cv2.fitEllipse(contour)
                    cv2.ellipse(frame, ellipse, (255, 0, 255), 2)
                    ecenter = ellipse[0]
                    cv2.circle(frame, tuple(map(int, ecenter)),
                               5, (255, 165, 0), -1)
                    ckey['center'] = np.array(ecenter, dtype=np.float32)
                    ckey['major_r'] = max(ellipse[1])
                self.display_frame_number(frame, frame_number, n_frames)
                cv2.imshow('Sauron', frame)
                if (cv2.waitKey(5) & 0xFF) == ord('q'):
                    break
            frame_number += 1
        cap.release()
        cv2.destroyAllWindows()

        self.insert1(key)
        for ckey in tqdm(contours):
            self.Ellipse().insert1(ckey, ignore_extra_fields=True)

# If config.yaml ever updated, make sure you store the file name differently so that it becomes unique
@schema
class ConfigDeeplabcut(dj.Manual):
    definition = """
    # Minimal info needed to load deeplabcut model
    config_path         : varchar(255)          # path to deeplabcut config
    ---
    shuffle             : smallint unsigned     # shuffle number used for the trained dlc model. Needed for dlc.analyze_videos
    trainingsetindex    : smallint unsigned     # trainingset index used for the trained dlc. model. Needed for dlc.analyze_videos
    """

@schema
class TrackingMethod(dj.Manual):
    definition="""
    -> Eye
    tracking_method     : tinyint unsigned # tracking_method 0:manual 1: deeplabcut
    ---
    """

    def fill(self, key, tracking_method):
        self.insert1(dict(key, tracking_method=tracking_method))

@schema
class Tracking(dj.Computed):
    definition="""
    -> TrackingMethod
    ---
    tracking_ts=CURRENT_TIMESTAMP   : timestamp  # automatic
    """

    class ManualTracking(dj.Part):
        definition="""
        -> master
        frame_id                    : int                   # frame id with matlab based 1 indexing
        ---
        contour=NULL                : longblob              # eye contour relative to ROI
        """
        def make(self, key):
            print("""
            This method is deprecated!
            Simply re-inserting data that were manually tracked before (ManuallyTrackedContours)!
            If data that matches with key does not exist, it will not be stored!
            """)

            if (ManuallyTrackedContours() & key).fetch1() is not None:
                for frame_id in range((ManuallyTrackedContours.Frame & key).fetch('frame_id').max()):
                    ckey = (ManuallyTrackedContours.Frame & dict(key, frame_id=frame_id)).fetch1()
                    self.insert1(dict(ckey, tracking_method= key['tracking_method']))
            else:
                print("Given key does not exist in ManuallyTrackedContours table!")
                print("Either manually track by populating ManuallyTrackedContours or use deeplabcut method!")

    class Deeplabcut(dj.Part):
        definition="""
        -> master
        ---
        short_vid_starting_index    : int unsigned          # middle frame index of the original video
        cropped_x0                  : smallint unsigned     # start width coord wrt original video
        cropped_x1                  : smallint unsigned     # end width coord wrt original video
        cropped_y0                  : smallint unsigned     # start height coord wrt original video
        cropped_y1                  : smallint unsigned     # end height coord wrt original video
        added_pixels                : smallint unsigned     # number of pixels added around the cropping coords
        config_path                 : varchar(128)          # path to deeplabcut config yaml        
        """

        def get_video_path(self, key):
            """
            Input:
                key: dictionary
                    A key that consists of animal_id, session, and scan_idx
            """
            video_info = (experiment.Session() *
                        experiment.Scan.EyeVideo() & key).fetch1()
            video_path = lab.Paths().get_local_path(
                "{behavior_path}/{filename}".format(**video_info))
            return video_path

        def create_tracking_directory(self, key):
            """
            this function creates the following directory structure:

            video_original_dir
                |
                |------ video_original
                |------ tracking_dir (create_tracking_directory)
                            |------- symlink to video_original (add_symlink)
                            |------- compressed_cropped_dir
                                        |------- cropped_video (generated by make_compressed_cropped_video function)
                                        |------- h5 file for cropped video (generated by deeplabcut)
                                        |------- pickle for cropped video (generated by deeplabcut)
                            |------- short_dir
                                        |------- short_video (generated by make_short_video function)
                                        |------- h5 file for short video(generated by deeplabcut)
                                        |------- pickle for short video (generated by deeplabcut)

            Input:
                key: dictionary
                    a dictionary that contains mouse id, session, and scan idx.

            Return:
                tracking_dir: string
                    a string that specifies the path to the tracking directory
            """

            print("Generating tracking directory for ", key)

            vid_path = self.get_video_path(key)
            vid_dir = os.path.dirname(os.path.normpath(vid_path))
            tracking_dir_name = os.path.basename(
                os.path.normpath(vid_path)).split('.')[0] + '_tracking'

            tracking_dir = os.path.join(vid_dir, tracking_dir_name)

            symlink_path = os.path.join(
                tracking_dir, os.path.basename(os.path.normpath(vid_path)))

            if not os.path.exists(tracking_dir):

                os.mkdir(tracking_dir)
                os.mkdir(os.path.join(tracking_dir, 'compressed_cropped'))
                os.mkdir(os.path.join(tracking_dir, 'short'))

                os.symlink(vid_path, symlink_path)

            else:
                print('{} already exists!'.format(tracking_dir))

            return tracking_dir, symlink_path

        def make_short_video(self, tracking_dir):
            """
            Extract 5 seconds long video starting from the middle of the original video.

            Input:
                tracking_dir: string
                    String that specifies the full path of tracking directory
            Return:
                None
            """
            from subprocess import Popen, PIPE

            suffix = '_short.avi'

            case = os.path.basename(os.path.normpath(
                tracking_dir)).split('_tracking')[0]

            input_video_path = os.path.join(tracking_dir, case + '.avi')

            out_vid_path = os.path.join(tracking_dir, 'short', case + suffix)

            cap = cv2.VideoCapture(input_video_path)

            original_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            original_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

            fps = cap.get(cv2.CAP_PROP_FPS)
            mid_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)/2)
            duration = int(mid_frame_num/fps)

            minutes, seconds = divmod(duration, 60)
            hours, minutes = divmod(minutes, 60)

            print('\nMaking a short video!')

            cmd = ['ffmpeg', '-i', input_video_path, '-ss',
                '{}:{}:{}'.format(hours, minutes, seconds), '-t', '5', '-c', 'copy', out_vid_path]

            # call ffmpeg to make a short video
            p = Popen(cmd, stdin=PIPE)
            # close ffmpeg
            p.wait()

            print('\nSuccessfully created a short video!')

            return out_vid_path, original_width, original_height, mid_frame_num

        def predict_labels(self, vid_path, config):
            """
            Predict labels on a given video

            Input:
                vid_path: string
                    Path to video.
                config: dictionary
                    a deeplabcut model configuration dictionary.
            """
            destfolder = os.path.dirname(vid_path)
            dlc.analyze_videos(config=config['config_path'], videos=[vid_path], videotype='avi', shuffle=config['shuffle'],
                            trainingsetindex=config['trainingsetindex'], gputouse=gputouse, save_as_csv=False, destfolder=destfolder)

        def obtain_cropping_coords(self, short_h5_path, DLCscorer, config):
            """
            First, filter out by the pcutoff, then find values that are within 1 std from mean
            for each eyelid bodypart. Then, compare among the parts and find min,max values in x and y.

            The reason we use 1 std from mean is that dlc might have outliers in this short video.
            Hence we filter out these potential outliers

            Input:
                short_h5_path: string
                    path to h5 file generated by deeplabcut on short video.
                DLCscorer: string
                    scorer name used for deeplabcut. Can be obtained via auxiliaryfunctions.GetScorerName(config, shuffle, trainsetindex)
                config: dictionary
                    a deeplabcut model configuration dictionary.
            """

            # there should be only 1 h5 file generated by dlc
            df_short = pd.read_hdf(short_h5_path)

            eyelid_cols = ['eyelid_top', 'eyelid_right',
                        'eyelid_left', 'eyelid_bottom']

            df_eyelid = df_short[DLCscorer][eyelid_cols]

            df_eyelid_likelihood = df_eyelid.iloc[:, df_eyelid.columns.get_level_values(
                1) == 'likelihood']
            df_eyelid_x = df_eyelid.iloc[:, df_eyelid.columns.get_level_values(
                1) == 'x']
            df_eyelid_y = df_eyelid.iloc[:, df_eyelid.columns.get_level_values(
                1) == 'y']

            df_eyelid_coord = dict(x=df_eyelid_x, y=df_eyelid_y)

            coords_dict = dict(xmin=[], xmax=[], ymin=[], ymax=[])

            for eyelid_label in eyelid_cols:

                for coord in ['x', 'y']:

                    # only obtain if the labels are confident enough (i.e. > pcutoff)
                    eyelid_coord_pcutoff = df_eyelid_coord[coord][(
                        df_eyelid_likelihood.loc[:, eyelid_label].values > config['pcutoff'])][eyelid_label][coord].values

                    eyelid_coord_68 = eyelid_coord_pcutoff[(eyelid_coord_pcutoff < np.mean(eyelid_coord_pcutoff) + np.std(eyelid_coord_pcutoff)) *
                                                        (eyelid_coord_pcutoff > np.mean(
                                                            eyelid_coord_pcutoff) - np.std(eyelid_coord_pcutoff))]

                    coords_dict[coord+'min'].append(eyelid_coord_68.min())
                    coords_dict[coord+'max'].append(eyelid_coord_68.max())

            cropped_coords = {}
            cropped_coords['cropped_x0'] = int(min(coords_dict['xmin']))
            cropped_coords['cropped_x1'] = int(max(coords_dict['xmax']))
            cropped_coords['cropped_y0'] = int(min(coords_dict['ymin']))
            cropped_coords['cropped_y1'] = int(max(coords_dict['ymax']))

            return cropped_coords

        def add_pixels(self, cropped_coords, original_width, original_height, pixel_num):
            """
            Add addtional pixels around cropped_coords
            Input:
                cropped_coords: dictionary
                    cropoping coordinates specifying left_top  and bottom_right coords
                original_width: int
                    width of the original video
                original_height: int
                    height of the original video
                pixel_num: int
                    number of pixels to add around the cropped_coords
            Return:
                cropped_coords: dictionary
                    updated cropoping coordinates specifying left_top  and bottom_right coords
            """

            if cropped_coords['cropped_x0'] - pixel_num < 0:
                cropped_coords['cropped_x0'] = 0
            else:
                cropped_coords['cropped_x0'] -= pixel_num

            if cropped_coords['cropped_x1'] + pixel_num > original_width:
                cropped_coords['cropped_x1'] = original_width
            else:
                cropped_coords['cropped_x1'] += pixel_num

            if cropped_coords['cropped_y0'] - pixel_num < 0:
                cropped_coords['cropped_y0'] = 0
            else:
                cropped_coords['cropped_y0'] -= pixel_num

            if cropped_coords['cropped_y1'] + pixel_num > original_height:
                cropped_coords['cropped_y1'] = original_height
            else:
                cropped_coords['cropped_y1'] += pixel_num

            return cropped_coords

        def make_compressed_cropped_video(self, tracking_dir, cropped_coords):
            """
            Make a compressed and cropped video.

            Input:
                tracking_dir: string
                    String that specifies the full path of tracking directory
                cropped_coords: dictionary
                    cropoping coordinates specifying left_top  and bottom_right coords
            Return:
                None
            """
            from subprocess import Popen, PIPE

            suffix = '_compressed_cropped.avi'

            case = os.path.basename(os.path.normpath(
                tracking_dir)).split('_tracking')[0]

            input_video_path = os.path.join(tracking_dir, case + '.avi')

            out_vid_path = os.path.join(
                tracking_dir, 'compressed_cropped', case + suffix)

            out_w = cropped_coords['cropped_x1'] - cropped_coords['cropped_x0']
            out_h = cropped_coords['cropped_y1'] - cropped_coords['cropped_y0']
            print('\nMaking a compressed and cropped video!')

            # crf: use value btw 17 and 28 (lower the number, higher the quality of the video)
            # intra: no compressing over time. only over space
            cmd = ['ffmpeg', '-i', '{}'.format(input_video_path), '-vcodec', 'libx264', '-crf', '17', '-intra', '-filter:v',
                "crop={}:{}:{}:{}".format(out_w, out_h, cropped_coords['cropped_x0'], cropped_coords['cropped_y0']), '{}'.format(out_vid_path)]

            # call ffmpeg to make a short video
            p = Popen(cmd, stdin=PIPE)
            # close ffmpeg
            p.wait()
            print('\nSuccessfully created a compressed & cropped video!\n')

            return out_vid_path

        def make(self, key):
            """
            Use Deeplabcut to label pupil and eyelids
            """
            # Disable DLC GUI first, then import deeplabcut
            os.environ["DLClight"] = "True"
            import deeplabcut as dlc
            from deeplabcut.utils import auxiliaryfunctions
            from .utils import DLC_tools

            print('Tracking labels with Deeplabcut!')

            # change config_path if we were to update DLC model configuration
            temp_config = (ConfigDeeplabcut & dict(
                config_path='/mnt/lab/DeepLabCut/pupil_track-Donnie-2019-02-12/config.yaml')).fetch1()
            
            # save config_path into the key
            key['config_path'] = temp_config['config_path']
            
            config = auxiliaryfunctions.read_config(temp_config['config_path'])
            config['config_path'] = temp_config['config_path']
            config['shuffle'] = temp_config['shuffle']
            config['trainingsetindex'] = temp_config['trainingsetindex']

            trainFraction = config['TrainingFraction'][config['trainingsetindex']]
            DLCscorer = auxiliaryfunctions.GetScorerName(
                config, config['shuffle'], trainFraction)

            # make needed directories
            tracking_dir, _ = self.create_tracking_directory(key)

            # make a short video (5 seconds long)
            short_video_path, original_width, original_height, mid_frame_num = self.make_short_video(
                tracking_dir)

            # save info about short video
            key['short_vid_starting_index'] = mid_frame_num
            
            short_h5_path = short_video_path.split('.')[0] + DLCscorer + '.h5'

            # predict using the short video
            self.predict_labels(short_video_path, config)

            # obtain the cropping coordinates from the prediciton on short video
            cropped_coords = self.obtain_cropping_coords(
                short_h5_path, DLCscorer, config)

            # add 100 pixels around cropping coords. Ensure that it is within the original dim
            pixel_num = 100
            cropped_coords = self.add_pixels(cropped_coords=cropped_coords,
                                             original_width=original_width,
                                             original_height=original_height,
                                             pixel_num=pixel_num)

            # make a compressed and cropped video
            compressed_cropped_video_path = self.make_compressed_cropped_video(
                tracking_dir, cropped_coords)

            # predict using the compressed and cropped video
            self.predict_labels(compressed_cropped_video_path, config)

            key = dict(key, cropped_x0=cropped_coords['cropped_x0'],
                            cropped_x1=cropped_coords['cropped_x1'],
                            cropped_y0=cropped_coords['cropped_y0'],
                            cropped_y1=cropped_coords['cropped_y1'],
                            added_pixels=pixel_num)

            self.insert1(key)

    def make(self, key):
        print("Tracking for case {}".format(key))

        if key['tracking_method'] == 0:
            self.insert1(key)
            self.ManualTracking().make(key)
        elif key['tracking_method'] == 1:
            self.insert1(key)
            self.Deeplabcut().make(key)
        else:
            msg = 'Unrecognized Tracking method {}'.format(key['tracking_method'])
            raise PipelineException(msg)
        
        
@schema
class AutomaticallyTrackedLabels(dj.Computed):
    definition = """
    # Tracking table using deeplabcut
    -> Eye
    method                                  : varchar(128)          # tracking method
    ---
    tracking_ts=CURRENT_TIMESTAMP           : timestamp             # automatic
    tracking_dir                            : varchar(255)          # path to tracking directory
    """

    class OriginalVideo(dj.Part):
        definition = """
        # original video information
        -> master
        ---
        original_width      : smallint unsigned    # original video width size
        original_height     : smallint unsigned    # original video height size
        video_path          : varchar(255)         # path to original video
        """

    class ShortVideo(dj.Part):
        definition = """
        # 5 seconds long video starting from the middle frame of the original video
        -> master
        ---
        starting_frame       : int unsigned        # middle frame of the original video
        video_path           : varchar(255)        # path to short video
        """

    class CompressedCroppedVideo(dj.Part):
        definition = """
        # Compressed and cropped video information
        -> master
        ---
        cropped_x0      : smallint unsigned        # start width coord wrt original video
        cropped_x1      : smallint unsigned        # end width coord wrt original video
        cropped_y0      : smallint unsigned        # start height coord wrt original video
        cropped_y1      : smallint unsigned        # end height coord wrt original video
        added_pixels    : smallint unsigned        # number of pixels added around the cropping coords
        video_path      : varchar(255)             # path to comparessed & cropped video
        """

    def get_video_path(self, key):
        """
        Input:
            key: dictionary
                A key that consists of animal_id, session, and scan_idx
        """
        video_info = (experiment.Session() *
                      experiment.Scan.EyeVideo() & key).fetch1()
        video_path = lab.Paths().get_local_path(
            "{behavior_path}/{filename}".format(**video_info))
        return video_path

    def create_tracking_directory(self, key):
        """
        this function creates the following directory structure:

        video_original_dir
            |
            |------ video_original
            |------ tracking_dir (create_tracking_directory)
                        |------- symlink to video_original (add_symlink)
                        |------- compressed_cropped_dir
                                    |------- cropped_video (generated by make_compressed_cropped_video function)
                                    |------- h5 file for cropped video (generated by deeplabcut)
                                    |------- pickle for cropped video (generated by deeplabcut)
                        |------- short_dir
                                    |------- short_video (generated by make_short_video function)
                                    |------- h5 file for short video(generated by deeplabcut)
                                    |------- pickle for short video (generated by deeplabcut)

        Input:
            key: dictionary
                a dictionary that contains mouse id, session, and scan idx.

        Return:
            tracking_dir: string
                a string that specifies the path to the tracking directory
        """

        print("Generating tracking directory for ", key)

        vid_path = self.get_video_path(key)
        vid_dir = os.path.dirname(os.path.normpath(vid_path))
        tracking_dir_name = os.path.basename(
            os.path.normpath(vid_path)).split('.')[0] + '_tracking'

        tracking_dir = os.path.join(vid_dir, tracking_dir_name)

        symlink_path = os.path.join(
            tracking_dir, os.path.basename(os.path.normpath(vid_path)))

        if not os.path.exists(tracking_dir):

            os.mkdir(tracking_dir)
            os.mkdir(os.path.join(tracking_dir, 'compressed_cropped'))
            os.mkdir(os.path.join(tracking_dir, 'short'))

            os.symlink(vid_path, symlink_path)

        else:
            print('{} already exists!'.format(tracking_dir))

        return tracking_dir, symlink_path

    def make_short_video(self, tracking_dir):
        """
        Extract 5 seconds long video starting from the middle of the original video.

        Input:
            tracking_dir: string
                String that specifies the full path of tracking directory
        Return:
            None
        """
        from subprocess import Popen, PIPE

        suffix = '_short.avi'

        case = os.path.basename(os.path.normpath(
            tracking_dir)).split('_tracking')[0]

        input_video_path = os.path.join(tracking_dir, case + '.avi')

        out_vid_path = os.path.join(tracking_dir, 'short', case + suffix)

        cap = cv2.VideoCapture(input_video_path)

        original_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        original_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        fps = cap.get(cv2.CAP_PROP_FPS)
        mid_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)/2)
        duration = int(mid_frame_num/fps)

        minutes, seconds = divmod(duration, 60)
        hours, minutes = divmod(minutes, 60)

        print('\nMaking a short video!')

        cmd = ['ffmpeg', '-i', input_video_path, '-ss',
               '{}:{}:{}'.format(hours, minutes, seconds), '-t', '5', '-c', 'copy', out_vid_path]

        # call ffmpeg to make a short video
        p = Popen(cmd, stdin=PIPE)
        # close ffmpeg
        p.wait()

        print('\nSuccessfully created a short video!')

        return out_vid_path, original_width, original_height, mid_frame_num

    def predict_labels(self, vid_path, config):
        """
        Predict labels on a given video

        Input:
            vid_path: string
                Path to video.
            config: dictionary
                a deeplabcut model configuration dictionary.
        """
        destfolder = os.path.dirname(vid_path)
        dlc.analyze_videos(config=config['config_path'], videos=[vid_path], videotype='avi', shuffle=config['shuffle'],
                           trainingsetindex=config['trainingsetindex'], gputouse=gputouse, save_as_csv=False, destfolder=destfolder)

    def obtain_cropping_coords(self, short_h5_path, DLCscorer, config):
        """
        First, filter out by the pcutoff, then find values that are within 1 std from mean
        for each eyelid bodypart. Then, compare among the parts and find min,max values in x and y.

        The reason we use 1 std from mean is that dlc might have outliers in this short video.
        Hence we filter out these potential outliers

        Input:
            short_h5_path: string
                path to h5 file generated by deeplabcut on short video.
            DLCscorer: string
                scorer name used for deeplabcut. Can be obtained via auxiliaryfunctions.GetScorerName(config, shuffle, trainsetindex)
            config: dictionary
                a deeplabcut model configuration dictionary.
        """

        # there should be only 1 h5 file generated by dlc
        df_short = pd.read_hdf(short_h5_path)

        eyelid_cols = ['eyelid_top', 'eyelid_right',
                       'eyelid_left', 'eyelid_bottom']

        df_eyelid = df_short[DLCscorer][eyelid_cols]

        df_eyelid_likelihood = df_eyelid.iloc[:, df_eyelid.columns.get_level_values(
            1) == 'likelihood']
        df_eyelid_x = df_eyelid.iloc[:, df_eyelid.columns.get_level_values(
            1) == 'x']
        df_eyelid_y = df_eyelid.iloc[:, df_eyelid.columns.get_level_values(
            1) == 'y']

        df_eyelid_coord = dict(x=df_eyelid_x, y=df_eyelid_y)

        coords_dict = dict(xmin=[], xmax=[], ymin=[], ymax=[])

        for eyelid_label in eyelid_cols:

            for coord in ['x', 'y']:

                # only obtain if the labels are confident enough (i.e. > pcutoff)
                eyelid_coord_pcutoff = df_eyelid_coord[coord][(
                    df_eyelid_likelihood.loc[:, eyelid_label].values > config['pcutoff'])][eyelid_label][coord].values

                eyelid_coord_68 = eyelid_coord_pcutoff[(eyelid_coord_pcutoff < np.mean(eyelid_coord_pcutoff) + np.std(eyelid_coord_pcutoff)) *
                                                       (eyelid_coord_pcutoff > np.mean(
                                                           eyelid_coord_pcutoff) - np.std(eyelid_coord_pcutoff))]

                coords_dict[coord+'min'].append(eyelid_coord_68.min())
                coords_dict[coord+'max'].append(eyelid_coord_68.max())

        cropped_coords = {}
        cropped_coords['cropped_x0'] = int(min(coords_dict['xmin']))
        cropped_coords['cropped_x1'] = int(max(coords_dict['xmax']))
        cropped_coords['cropped_y0'] = int(min(coords_dict['ymin']))
        cropped_coords['cropped_y1'] = int(max(coords_dict['ymax']))

        return cropped_coords

    def add_pixels(self, cropped_coords, original_width, original_height, pixel_num):
        """
        Add addtional pixels around cropped_coords
        Input:
            cropped_coords: dictionary
                cropoping coordinates specifying left_top  and bottom_right coords
            original_width: int
                width of the original video
            original_height: int
                height of the original video
            pixel_num: int
                number of pixels to add around the cropped_coords
        Return:
            cropped_coords: dictionary
                updated cropoping coordinates specifying left_top  and bottom_right coords
        """

        if cropped_coords['cropped_x0'] - pixel_num < 0:
            cropped_coords['cropped_x0'] = 0
        else:
            cropped_coords['cropped_x0'] -= pixel_num

        if cropped_coords['cropped_x1'] + pixel_num > original_width:
            cropped_coords['cropped_x1'] = original_width
        else:
            cropped_coords['cropped_x1'] += pixel_num

        if cropped_coords['cropped_y0'] - pixel_num < 0:
            cropped_coords['cropped_y0'] = 0
        else:
            cropped_coords['cropped_y0'] -= pixel_num

        if cropped_coords['cropped_y1'] + pixel_num > original_height:
            cropped_coords['cropped_y1'] = original_height
        else:
            cropped_coords['cropped_y1'] += pixel_num

        return cropped_coords

    def make_compressed_cropped_video(self, tracking_dir, cropped_coords):
        """
        Make a compressed and cropped video.

        Input:
            tracking_dir: string
                String that specifies the full path of tracking directory
            cropped_coords: dictionary
                cropoping coordinates specifying left_top  and bottom_right coords
        Return:
            None
        """
        from subprocess import Popen, PIPE

        suffix = '_compressed_cropped.avi'

        case = os.path.basename(os.path.normpath(
            tracking_dir)).split('_tracking')[0]

        input_video_path = os.path.join(tracking_dir, case + '.avi')

        out_vid_path = os.path.join(
            tracking_dir, 'compressed_cropped', case + suffix)

        out_w = cropped_coords['cropped_x1'] - cropped_coords['cropped_x0']
        out_h = cropped_coords['cropped_y1'] - cropped_coords['cropped_y0']
        print('\nMaking a compressed and cropped video!')

        # crf: use value btw 17 and 28 (lower the number, higher the quality of the video)
        # intra: no compressing over time. only over space
        cmd = ['ffmpeg', '-i', '{}'.format(input_video_path), '-vcodec', 'libx264', '-crf', '17', '-intra', '-filter:v',
               "crop={}:{}:{}:{}".format(out_w, out_h, cropped_coords['cropped_x0'], cropped_coords['cropped_y0']), '{}'.format(out_vid_path)]

        # call ffmpeg to make a short video
        p = Popen(cmd, stdin=PIPE)
        # close ffmpeg
        p.wait()
        print('\nSuccessfully created a compressed & cropped video!\n')

        return out_vid_path

    def make(self, key):
        """
        Automatically predict the pupil labels

        Args:
            key (dict): contains animal_id, session, scan_idx, and method.
                        The only available method is deeplabcut at the moment.
        Returns:
            None
        """
        if key['method'].lower() in ['deeplabcut', 'dlc']:
            print('Tracking labels with DLC')

            # change config_path if we were to update DLC model configuration
            temp_config = (ConfigDeeplabcut & dict(
                config_path='/mnt/lab/DeepLabCut/pupil_track-Donnie-2019-02-12/config.yaml')).fetch1()
            config = auxiliaryfunctions.read_config(temp_config['config_path'])
            config['config_path'] = temp_config['config_path']
            config['shuffle'] = temp_config['shuffle']
            config['trainingsetindex'] = temp_config['trainingsetindex']

            trainFraction = config['TrainingFraction'][config['trainingsetindex']]
            DLCscorer = auxiliaryfunctions.GetScorerName(
                config, config['shuffle'], trainFraction)

            # make needed directories
            tracking_dir, original_video_path = self.create_tracking_directory(key)
            self.insert1(dict(key, tracking_dir=tracking_dir))

            # make a short video (5 seconds long)
            short_video_path, original_width, original_height, mid_frame_num = self.make_short_video(
                tracking_dir)

            # save info about original video
            original_video = self.OriginalVideo()
            original_video.insert1(
                dict(key, original_width=original_width,
                     original_height=original_height,
                     video_path=original_video_path))

            # save info about short video
            short_video = self.ShortVideo()
            short_video.insert1(
                dict(key, starting_frame=mid_frame_num, video_path=short_video_path))

            short_h5_path = short_video_path.split('.')[0] + DLCscorer + '.h5'

            # predict using the short video
            self.predict_labels(short_video_path, config)

            # obtain the cropping coordinates from the prediciton on short video
            cropped_coords = self.obtain_cropping_coords(
                short_h5_path, DLCscorer, config)

            # add 100 pixels around cropping coords. Ensure that it is within the original dim
            pixel_num = 100
            cropped_coords = self.add_pixels(cropped_coords=cropped_coords,
                                             original_width=original_width,
                                             original_height=original_height,
                                             pixel_num=pixel_num)

            # make a compressed and cropped video
            compressed_cropped_video_path = self.make_compressed_cropped_video(
                tracking_dir, cropped_coords)

            # predict using the compressed and cropped video
            self.predict_labels(compressed_cropped_video_path, config)

            compressed_cropped_video = self.CompressedCroppedVideo()
            compressed_cropped_video.insert1(dict(key, cropped_x0=cropped_coords['cropped_x0'],
                                                  cropped_x1=cropped_coords['cropped_x1'],
                                                  cropped_y0=cropped_coords['cropped_y0'],
                                                  cropped_y1=cropped_coords['cropped_y1'],
                                                  added_pixels=pixel_num,
                                                  video_path=compressed_cropped_video_path))

@schema
class FittedContourNew(dj.Computed):
    definition="""
    # Fit a circle and an ellipse
    -> Tracking
    ---
    fitting_ts=CURRENT_TIMESTAMP    : timestamp     # automatic
    """

    class Circle(dj.Part):
        definition = """
        -> master
        frame_id                 : int           # frame id with matlab based 1 indexing
        ---
        center=NULL              : tinyblob      # center of the circle in (x, y) of image
        radius=NULL              : float         # radius of the circle
        visible_portion=NULL     : float         # portion of visible pupil area given a fitted circle frame. Please refer DLC_tools.PupilFitting.detect_visible_pupil_area for more details

        """

    class Ellipse(dj.Part):
        definition = """
        -> master
        frame_id                 : int           # frame id with matlab based 1 indexing
        ---
        center=NULL              : tinyblob      # center of the ellipse in (x, y) of image
        major_radius=NULL        : float         # major radius of the ellipse
        minor_radius=NULL        : float         # minor radius of the ellipse
        rotation_angle=NULL      : float         # ellipse rotation angle in degrees w.r.t. major_radius
        visible_portion=NULL     : float         # portion of visible pupil area given a fitted ellipse frame. Please refer DLC_tools.PupilFitting.detect_visible_pupil_area for more details
        """

    def make(self, key):
        print("Fitting:", key)

        self.insert1(key)

        # manual == 0
        if key['tracking_method'] == 0:

            avi_path = (Eye() & key).get_video_path()

            contours = (Tracking.ManualTracking() & key).fetch(
                order_by='frame_id ASC', as_dict=True)
            
            video = DLC_tools.video_processor.VideoProcessorCV(fname=avi_path)

            # for manual tracking, we did not track eyelids, hence put -1.0 to be 
            # consistent with how we defined under PupilFitting.detect_visible_pupil_area
            visible_portion = -1.0 
            for frame_num in tdqm(range(video.nframes)):
                ckey = contours[frame_num]

                if ckey['contour'] is not None:

                    # fit circle. This is consistent with fitting method for DLC
                    if len(ckey['contour']) >= 3:
                        x, y, radius = DLC_tools.smallest_enclosing_circle_naive(ckey['contour'])
                        center = (x, y)
                        self.Circle().insert1(dict(key, frame_id=frame_num,
                                            center=center,
                                            radius=radius,
                                            visible_portion=visible_portion))
                    else:
                        # if less than 3, then we do not have enough pupil labels nor
                        # we have eyelid labels
                        self.Circle().insert1(dict(key, frame_id=frame_num,
                                            center=None,
                                            radius=None,
                                            visible_portion=-3.0))   

                    # fit ellipse. This is consistent with fitting method for DLC
                    if len(ckey['contour']) >= 6:
                        rotated_rect = cv2.fitEllipse(ckey['contour'])
                        self.Ellipse().insert1(dict(key, frame_id=frame_num,
                                            center=rotated_rect[0],
                                            major_radius=rotated_rect[1][1]/2.0,
                                            minor_radius=rotated_rect[1][0]/2.0,
                                            rotation_angle=rotated_rect[2],
                                            visible_portion=visible_portion))
                    else:
                        # if less than 3, then we do not have enough pupil labels nor
                        # we have eyelid labels
                        self.Ellipse().insert1(dict(key, frame_id=frame_num, 
                                            center=None,
                                            major_radius=None,
                                            minor_radius=None,
                                            rotation_angle=None,
                                            visible_portion=-3.0))

        elif key['tracking_method'] == 1:

            dlc_config = (ConfigDeeplabcut & (Tracking.Deeplabcut & key)).fetch1()

            config = auxiliaryfunctions.read_config(dlc_config['config_path'])
            config['config_path'] = dlc_config['config_path']
            config['shuffle'] = dlc_config['shuffle']
            config['trainingsetindex'] = dlc_config['trainingsetindex']

            # find path to compressed_cropped_video
            base_path = os.path.splitext((Eye() & key).get_video_path())[0] + '_tracking'
            
            for root, _, files in os.walk(base_path):
                for file in files:
                    if file.endswith('compressed_cropped.avi'):
                        cc_vid_path = os.path.join(root, file)

            config['video_path'] = cc_vid_path

            pupil_fit = DLC_tools.PupilFitting(config=config, bodyparts='all')

            for frame_num in tqdm(range(pupil_fit.clip.nframes)):

                fit_dict = pupil_fit.fitted_core(frame_num=frame_num)

                self.Circle().insert1(dict(key, frame_id=frame_num,
                                    center=fit_dict['circle_fit']['center'],
                                    radius=fit_dict['circle_fit']['radius'],
                                    visible_portion=fit_dict['circle_visible']['visible_portion']))

                self.Ellipse().insert1(dict(key, frame_id=frame_num,
                                    center=fit_dict['ellipse_fit']['center'],
                                    major_radius=fit_dict['ellipse_fit']['major_radius'],
                                    minor_radius=fit_dict['ellipse_fit']['minor_radius'],
                                    rotation_angle=fit_dict['ellipse_fit']['rotation_angle'],
                                    visible_portion=fit_dict['ellipse_visible']['visible_portion']))
        
        
@schema
class FittedContourDeeplabcut(dj.Computed):
    definition = """
    # Fit a circle and an ellipse using compressed & cropped video.
    -> TrackedLabelsDeeplabcut
    ---
    fitting_ts=CURRENT_TIMESTAMP    : timestamp  # automatic
    """

    class Circle(dj.Part):
        definition = """
        -> master
        frame_id                 : int           # frame id with matlab based 1 indexing
        ---
        center=NULL              : tinyblob      # center of the circle in (x, y) of image
        radius=NULL              : float         # radius of the circle
        visible_portion=NULL     : float         # portion of visible pupil area given a fitted circle frame. Please refer DLC_tools.PupilFitting.detect_visible_pupil_area for more details

        """

    class Ellipse(dj.Part):
        definition = """
        -> master
        frame_id                 : int           # frame id with matlab based 1 indexing
        ---
        center=NULL              : tinyblob      # center of the ellipse in (x, y) of image
        major_radius=NULL        : float         # major radius of the ellipse
        minor_radius=NULL        : float         # minor radius of the ellipse
        rotation_angle=NULL      : float         # ellipse rotation angle in degrees w.r.t. major_radius
        visible_portion=NULL     : float         # portion of visible pupil area given a fitted ellipse frame. Please refer DLC_tools.PupilFitting.detect_visible_pupil_area for more details
        """

    def make(self, key):
        print("Fitting:", key)

        tracking_info = (TrackedLabelsDeeplabcut & key).fetch1()
        shuffle, trainingsetindex = (ConfigDeeplabcut & tracking_info).fetch1(
            'shuffle', 'trainingsetindex')

        config = auxiliaryfunctions.read_config(tracking_info['config_path'])
        config['config_path'] = tracking_info['config_path']
        config['shuffle'] = shuffle
        config['trainingsetindex'] = trainingsetindex

        compressed_cropped_vid_path = (
            TrackedLabelsDeeplabcut.CompressedCroppedVideo & key).fetch1('video_path')

        config['video_path'] = compressed_cropped_vid_path

        pupil_fit = DLC_tools.PupilFitting(config=config, bodyparts='all')

        self.insert1(key)

        for frame_num in tqdm(range(pupil_fit.clip.nframes)):

            fit_dict = pupil_fit.fitted_core(frame_num=frame_num)

            circle = FittedContourDeeplabcut.Circle()
            circle.insert1(dict(key, frame_id=frame_num,
                                center=fit_dict['circle_fit']['center'],
                                radius=fit_dict['circle_fit']['radius'],
                                visible_portion=fit_dict['circle_visible']['visible_portion']))

            ellipse = FittedContourDeeplabcut.Ellipse()
            ellipse.insert1(dict(key, frame_id=frame_num,
                                 center=fit_dict['ellipse_fit']['center'],
                                 major_radius=fit_dict['ellipse_fit']['major_radius'],
                                 minor_radius=fit_dict['ellipse_fit']['minor_radius'],
                                 rotation_angle=fit_dict['ellipse_fit']['rotation_angle'],
                                 visible_portion=fit_dict['ellipse_visible']['visible_portion']))
