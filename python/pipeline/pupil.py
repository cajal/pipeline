"""
This module is for pupil related tables and computation.
Current version is at 1.0.0 as of 10/29/2019

What's changed in this version:
1. Got rid of filter_config table
2. Added a functionality of removing tracking directory and its sub-directories if the job was not done completely
3. Method name changed from DLC_tools.filter_by_std to DLC_tools.filter_by_fitting_std
4. Deleted OnlineMedianFilteredFittedPupil table (it is already dropped on DB side)
5. Added a missing schema decorator on SurfaceMethod table
6. Bug fixed when finding cropping coords from short video (DLC related bug) (commit a861127)
7. Projector and ProjectorDisplay tables added
"""

# Disable DLC GUI first, then import deeplabcut
import os
os.environ["DLClight"] = "True"

import deeplabcut as dlc
from .utils import DLC_tools
from deeplabcut.utils import auxiliaryfunctions
from itertools import count
from scipy.misc import imresize
from tqdm import tqdm
import shutil
import cv2
import numpy as np
import pandas as pd
import json
from commons import lab
import datajoint as dj
from datajoint.jobs import key_hash
from datajoint.autopopulate import AutoPopulate

from .utils.decorators import gitlog
from .utils import eye_tracking, h5
from .utils.eye_tracking import PupilTracker, ManualTracker
from . import config
from . import experiment, notify, shared
from .exceptions import PipelineException

from IPython import display
import pylab as pl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import time
import datetime


__VERSION__ = "1.0.0"

schema = dj.schema('pipeline_eye')


dj.config['external-pupil'] = dict(protocol='file',
       location='/mnt/dj-stor01/pupil_fitting')


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
    definition = """  
    # eye movie timestamps synchronized to behavior clock

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

    def make(self, key):
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
            else:  # fill with NaNs
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

    def make(self, key):
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

        msg = """
        ManuallyTrackedContours table is now deprecated! 
        If you wanna track manually, please use Tracking.ManualTracking table:
        pupil.Tracking.populate(key, 'tracking_method=1')
        """

        print(msg)


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
class Tracking(dj.Computed):
    definition = """
    -> Eye
    -> shared.TrackingMethod
    ---
    tracking_ts=CURRENT_TIMESTAMP   : timestamp  # automatic
    """

    class ManualTrackingParameter(dj.Part):
        definition = """
        -> master.ManualTracking
        ---
        min_lambda=NULL         : float     # minimum mixing weight for current frame in running average computation (1 means no running avg was used)
        roi=NULL                : longblob  # roi of eye
        gauss_blur=NULL         : float     # bluring of ROI
        exponent=NULL           : tinyint   # exponent for contrast enhancement
        dilation_iter=NULL      : tinyint   # number of dilation and erosion operations
        min_contour_len=NULL    : tinyint   # minimal contour length
        running_avg_mix=NULL    : float     # weight a in a * current_frame + (1-a) * running_avg
        """

    class ManualTracking(dj.Part):
        definition = """
        -> master
        frame_id                    : int                   # frame id (starting from 0)
        ---
        contour=NULL                : longblob              # eye contour relative to ROI
        """
        def make(self, key, backup_file=None):

            # key does exist in ManuallyTrackedContours (i.e. we tracked before)
            if len(ManuallyTrackedContours() & key) > 0:

                print("""
                The given key has been tracked manually before (from ManuallyTrackedContours)! 
                Simply re-inserting previously tracked data here!
                """)

                # check if parameter table was populated b4. 
                # If not, we can skip inserting param information
                if len(ManuallyTrackedContours.Parameter & key) == 0:
                    for frame_id in range(len(ManuallyTrackedContours.Frame & key)):
                        # copy Frame info
                        frame_key = (ManuallyTrackedContours.Frame &
                                    dict(key, frame_id=frame_id)).fetch1()
                        self.insert1(
                            dict(frame_key, tracking_method=key['tracking_method']))

                else:
                    for frame_id in range(len(ManuallyTrackedContours.Frame & key)):
                        # copy Frame info
                        frame_key = (ManuallyTrackedContours.Frame &
                                    dict(key, frame_id=frame_id)).fetch1()
                        self.insert1(
                            dict(frame_key, tracking_method=key['tracking_method']))
                        # copy Parameter info
                        param_key = (ManuallyTrackedContours.Parameter &
                                    dict(key, frame_id=frame_id)).fetch1()
                        min_lambda = (ManuallyTrackedContours &
                                    key).fetch1('min_lambda')
                        Tracking.ManualTrackingParameter.insert1(
                            dict(param_key, min_lambda=min_lambda, tracking_method=key['tracking_method']))

            # key does not exist in ManuallyTrackedContours, hence need to trace manually
            else:
                print("Manually Tracking!")

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
                min_lambda = logtrace[logtrace > 0].min()
                frame = Tracking.ManualTracking()
                parameters = Tracking.ManualTrackingParameter()
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
                        dict(key, **params, min_lambda=min_lambda), ignore_extra_fields=True)

    class Deeplabcut(dj.Part):
        definition = """
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

            hardlink_path = os.path.join(
                tracking_dir, os.path.basename(os.path.normpath(vid_path)))

            if not os.path.exists(tracking_dir):

                os.mkdir(tracking_dir)
                os.mkdir(os.path.join(tracking_dir, 'compressed_cropped'))
                os.mkdir(os.path.join(tracking_dir, 'short'))

                os.link(vid_path, hardlink_path)

            else:
                print('{} already exists!'.format(tracking_dir))
                print('Removing existing tracking directory and recreating')

                shutil.rmtree(tracking_dir)

                os.mkdir(tracking_dir)
                os.mkdir(os.path.join(tracking_dir, 'compressed_cropped'))
                os.mkdir(os.path.join(tracking_dir, 'short'))

                os.link(vid_path, hardlink_path)

            return tracking_dir, hardlink_path

        def make(self, key):
            """
            Use Deeplabcut to label pupil and eyelids
            """

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
            short_video_path, original_width, original_height, mid_frame_num = DLC_tools.make_short_video(
                tracking_dir)

            # add original width and height to config
            config['original_width'] = original_width
            config['original_height'] = original_height

            # save info about short video
            key['short_vid_starting_index'] = mid_frame_num

            short_h5_path = short_video_path.split('.')[0] + DLCscorer + '.h5'

            # predict using the short video
            DLC_tools.predict_labels(short_video_path, config)

            # obtain the cropping coordinates from the prediciton on short video
            cropped_coords = DLC_tools.obtain_cropping_coords(
                short_h5_path, DLCscorer, config)

            # add 100 pixels around cropping coords. Ensure that it is within the original dim
            pixel_num = 100
            cropped_coords = DLC_tools.add_pixels(cropped_coords=cropped_coords,
                                                  original_width=original_width,
                                                  original_height=original_height,
                                                  pixel_num=pixel_num)

            # make a compressed and cropped video
            compressed_cropped_video_path = DLC_tools.make_compressed_cropped_video(
                tracking_dir, cropped_coords)

            # predict using the compressed and cropped video
            DLC_tools.predict_labels(compressed_cropped_video_path, config)

            key = dict(key, cropped_x0=cropped_coords['cropped_x0'],
                       cropped_x1=cropped_coords['cropped_x1'],
                       cropped_y0=cropped_coords['cropped_y0'],
                       cropped_y1=cropped_coords['cropped_y1'],
                       added_pixels=pixel_num)

            self.insert1(key)

            # delete short video directory
            shutil.rmtree(os.path.dirname(short_video_path))

            # delete compressed and cropped video
            os.remove(compressed_cropped_video_path)

    def make(self, key):
        print("Tracking for case {}".format(key))

        if key['tracking_method'] == 1:
            self.insert1(key)
            self.ManualTracking().make(key)
        elif key['tracking_method'] == 2:
            self.insert1(key)
            self.Deeplabcut().make(key)
        else:
            msg = 'Unrecognized Tracking method {}'.format(
                key['tracking_method'])
            raise PipelineException(msg)


@schema
class FittedPupil(dj.Computed):
    definition = """
    # Fit a circle and an ellipse
    -> Tracking
    ---
    fitting_ts=CURRENT_TIMESTAMP    : timestamp     # automatic
    """
    
    class EyePoints(dj.Part):
        definition = """
        -> master
        label             : char(20)            # body part label for given points
        ---
        x                 : longblob            # array with the x coordinates of point_label in the eye video
        y                 : longblob            # array with the y coordinatess of point_label in the eye video
    
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

        avi_path = (Eye & key).get_video_path()
        nframes = (Eye & key).fetch1('total_frames')

        data_circle = []
        data_ellipse = []

        # manual == 1
        if key['tracking_method'] == 1:

            contours = (Tracking.ManualTracking & key).fetch(
                'contour', order_by='frame_id ASC')

            # for manual tracking, we did not track eyelids, hence put -1.0 to be
            # consistent with how we defined under PupilFitting.detect_visible_pupil_area
            visible_portion = -1.0

            for frame_num in tqdm(range(nframes)):

                if contours[frame_num] is None or len(contours[frame_num].squeeze()) < 3:

                    data_circle.append([None, None, -3.0])

                if contours[frame_num] is None or len(contours[frame_num].squeeze()) < 6:

                    data_ellipse.append([None, None, None, None, -3.0])

                if contours[frame_num] is not None and len(contours[frame_num].squeeze()) >= 3:
                    x, y, radius = DLC_tools.smallest_enclosing_circle_naive(
                        contours[frame_num].squeeze())
                    center = np.array(x, y)

                    data_circle.append([center, radius, visible_portion])

                if contours[frame_num] is not None and len(contours[frame_num]) >= 6:
                    rotated_rect = cv2.fitEllipse(
                        contours[frame_num].squeeze())

                    data_ellipse.append([rotated_rect[0],
                                         rotated_rect[1][1] /
                                         2.0, rotated_rect[1][0]/2.0,
                                         rotated_rect[2], visible_portion])

        # deeplabcut 2
        elif key['tracking_method'] == 2:

            dlc_config = (ConfigDeeplabcut & (
                Tracking.Deeplabcut & key)).fetch1()

            config = auxiliaryfunctions.read_config(dlc_config['config_path'])
            config['config_path'] = dlc_config['config_path']
            config['shuffle'] = dlc_config['shuffle']
            config['trainingsetindex'] = dlc_config['trainingsetindex']

            # find path to original video symlink
            base_path = os.path.splitext(avi_path)[0] + '_tracking'
            video_path = os.path.join(base_path, os.path.basename(avi_path))

            config['orig_video_path'] = video_path

            # find croppoing coords
            cropped_coords = (Tracking.Deeplabcut & key).fetch1(
                'cropped_x0', 'cropped_x1', 'cropped_y0', 'cropped_y1')

            config['cropped_coords'] = cropped_coords

            pupil_fit = DLC_tools.DeeplabcutPupilFitting(
                config=config, bodyparts='all', cropped=True)

            for bodypart in pupil_fit.bodyparts:
                self.EyePoints.insert1({**key,
                    'label': bodypart,
                    'x': pupil_fit.df_bodyparts[bodypart]['x'].values,
                    'y': pupil_fit.df_bodyparts[bodypart]['y'].values       
                })

            for frame_num in tqdm(range(nframes)):

                fit_dict = pupil_fit.fitted_core(frame_num=frame_num)

                # circle info
                center = fit_dict['circle_fit']['center']
                radius = fit_dict['circle_fit']['radius']
                visible_portion = fit_dict['circle_visible']['visible_portion']

                data_circle.append([center, radius, visible_portion])

                # ellipse info
                center = fit_dict['ellipse_fit']['center']
                major_radius = fit_dict['ellipse_fit']['major_radius']
                minor_radius = fit_dict['ellipse_fit']['minor_radius']
                rotation_angle = fit_dict['ellipse_fit']['rotation_angle']
                visible_portion = fit_dict['ellipse_visible']['visible_portion']

                data_ellipse.append([center, major_radius, minor_radius,
                                     rotation_angle, visible_portion])

        data_circle = np.array(data_circle)
        data_ellipse = np.array(data_ellipse)

        # now filter out the outliers by 5.5 std away from mean
        rejected_ind = DLC_tools.filter_by_fitting_std(
            data=data_circle, fitting_method='circle', std_magnitude=5.5)

        data_circle[rejected_ind] = None, None, -3.0

        common_entry = np.array(list(key.values()))
        common_matrix = np.tile(common_entry, (nframes, 1))

        data_circle = np.hstack(
            (common_matrix, np.arange(nframes).reshape(-1, 1), data_circle))

        # insert data
        self.Circle.insert(data_circle)

        # now repeat the process for ellipse
        rejected_ind = DLC_tools.filter_by_fitting_std(
            data=data_ellipse, fitting_method='ellipse', std_magnitude=5.5)

        data_ellipse[rejected_ind, :] = None, None, None, None, -3.0

        data_ellipse = np.hstack(
            (common_matrix, np.arange(nframes).reshape(-1, 1), data_ellipse))

        self.Ellipse.insert(data_ellipse)



def plot_fitting(key, start, end=-1, fit_type='Circle', fig=None, ax=None, mask_flag=True):
    """Plot the fitted frame. Note this plotting method only works for Circle, not an ellipse

    Args:
        key (dict): A dictionary that contains animal_id, session, scan_idx, and tracking_method as keys
        start (int): A number indicating the start of the frame. 
            If only start is provided, then plot only one frame.
        end (int, optional): A number indicating the end of the frame. 
            If both start and end provided, then plot multiple frames. Otherwise, Default to -1.
        fit_type (str, optional): A string indicating what to plot. Default to 'Circle'. Other option is 'Ellipse'
        fig (:obj matplotlib.figure.Figure, optional): Figure object to pass. Default to None.
        ax (:obj matplotlib.axes._subplots.AxesSubplot, optional): Axes object to pass. Defualt to None.
        mask_flag (boolean, optional): Whether to show the visible area or not for DLC fitting. 
            Only relevant if tracking_method is 2. Default to True

    Returns:
        fig (:obj matplotlib.figure.Figure, optional): Figure object to pass. if fig provided as an argument,
            return the same fig object after updating
        ax (:obj matplotlib.axes._subplots.AxesSubplot, optional): Axes object to pass. if ax provided as an argument,
            return the same ax object after updating
    """
    from IPython import display
    import pylab as pl
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    import time

    if 'tracking_method' not in key.keys():
        raise KeyError('tracking_method is not define!')

    if end == -1:
        end = start + 1

    fit_type = fit_type.lower()
    if fit_type not in ['circle', 'ellipse']:
        raise ValueError('fit_type must be either a circle or an ellipse')

    # find croppoing coords
    # It is possible that the provided key was not tracked with DLC. Then cropped_coords is the same size as the original frame size
    if len(Tracking.Deeplabcut & dict(key, tracking_method=2)) == 0:
        cropped_coords = (0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 0, int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    elif len(Tracking.Deeplabcut & dict(key, tracking_method=2)) == 1:
        cropped_coords = (Tracking.Deeplabcut & dict(key, tracking_method=2)).fetch1(
            'cropped_x0', 'cropped_x1', 'cropped_y0', 'cropped_y1')
    else:
        raise ValueError(
            'Fitting is corrupted! Ensure that Tracking.Deeplabcut & dict(key, tracking_method=2) is unique!')

    # prepare plotting
    if ax is None:
        fig = plt.figure(frameon=False, figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1)
        plt.subplots_adjust(left=0, bottom=0, right=1,
                            top=1, wspace=0, hspace=0)

    # get video path
    avi_path = (Eye & key).get_video_path()
    cap = cv2.VideoCapture(avi_path)

    plt.xlim(0, cap.get(cv2.CAP_PROP_FRAME_WIDTH) - cropped_coords[0])
    plt.ylim(0, cap.get(cv2.CAP_PROP_FRAME_HEIGHT) - cropped_coords[2])

    plt.gca().invert_yaxis()

    if key['tracking_method'] == 1:

        if fit_type == 'circle':
            center, radius = (FittedPupil.Circle() & key & 'frame_id >= {}'.format(start) & 'frame_id < {}'.format(end+1)).fetch(
                'center', 'radius', order_by='frame_id')
        else:
            center, major_r, minor_r, angle = (FittedPupil.Ellipse() & key & 'frame_id >= {}'.format(start) & 'frame_id < {}'.format(end+1)).fetch(
                'center', 'major_radius', 'minor_radius', 'rotation_angle', order_by='frame_id')
        # manual
        contours = (Tracking.ManualTracking & key & 'frame_id >= {}'.format(start) & 'frame_id < {}'.format(end+1)).fetch(
            'contour', order_by='frame_id')

        for frame_num in range(start, end):

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            _, img = cap.read()
            img = np.flip(img, 2)[
                cropped_coords[2]: cropped_coords[3], cropped_coords[0]: cropped_coords[1]]

            ax_frame = ax.imshow(img, cmap='gray')

            if contours[frame_num-start] is not None:
                ind = frame_num-start

                if fit_type == 'circle':

                    plt_fit = plt.Circle((center[ind][0]-cropped_coords[0], center[ind][1]-cropped_coords[2]),
                                         radius[ind], color='b', fill=False)

                else:
                    plt_fit = Ellipse((center[ind][0]-cropped_coords[0], center[ind][1]-cropped_coords[2]),
                                      major_r[ind]*2, minor_r[ind]*2, angle[ind]-90, color='b', fill=False)

                ax.add_patch(plt_fit)
                ax_scatter = ax.scatter(contours[ind].squeeze()[:, 0] - cropped_coords[0],
                                        contours[ind].squeeze()[:, 1] -
                                        cropped_coords[2],
                                        s=4**2, color='red', alpha=.5)
                # ax.add_collection(ax_scatter)

            ax.axis('off')
            ax.set_title('frame num: ' + str(frame_num), fontsize=10)
            plt.tight_layout()

            fig.canvas.draw()

            # check if we plot only a single frame. If so, return fig and ax
            if end == start + 1:
                return fig, ax

            display.clear_output(wait=True)
            display.display(pl.gcf())
            time.sleep(0.5)

            plt.cla()
        plt.close('all')

    elif key['tracking_method'] == 2:
        # deeplabcut
        dlc_config = (ConfigDeeplabcut & (Tracking.Deeplabcut & key)).fetch1()

        config = auxiliaryfunctions.read_config(dlc_config['config_path'])
        config['config_path'] = dlc_config['config_path']
        config['shuffle'] = dlc_config['shuffle']
        config['trainingsetindex'] = dlc_config['trainingsetindex']
        config['cropped_coords'] = cropped_coords

        # find path to original video symlink
        base_path = os.path.splitext(avi_path)[0] + '_tracking'
        video_path = os.path.join(
            base_path, os.path.basename((Eye() & key).get_video_path()))

        config['orig_video_path'] = video_path

        pupil_fit = DLC_tools.DeeplabcutPupilFitting(
            config=config, bodyparts='all', cropped=True)

        # play with these parameters for better visualization of the fitting
        pupil_fit.line_thickness = 2
        pupil_fit.fontsize = 10
        pupil_fit.circle_color = (0, 0, 255)
        pupil_fit.ellipse_color = (0, 0, 255)
        pupil_fit.dotsize = 4

        # check if we plot only a single frame. If so, return fig and ax
        if end == start + 1:
            ax = pupil_fit.plot_fitted_frame(
                start, ax=ax, fitting_method=fit_type)

            # remove mask (i.e. visible area)
            if not mask_flag:
                del ax.images[1]

            return fig, ax

        else:
            pupil_fit.plot_fitted_multi_frames(
                start=start, end=end, fitting_method=fit_type)


# class PlotFitting(object):

#     def __init__(self, key):

#         if 'tracking_method' not in key.keys():
#             raise KeyError('tracking_method is not define!')

#         self.key = key
#         self.tracking_method = self.key['tracking_method']
        
#         # get video path
#         avi_path = (Eye & key).get_video_path()
#         cap = cv2.VideoCapture(avi_path)

#         self.avi_path = avi_path
#         self.cap = cap

#         # find croppoing coords
#         # It is possible that the provided key was not tracked with DLC. Then cropped_coords is the same size as the original frame size
#         if len(Tracking.Deeplabcut & dict(key, tracking_method=2)) == 0:
#             cropped_coords = (0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 0, int(
#                 cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#         elif len(Tracking.Deeplabcut & dict(key, tracking_method=2)) == 1:
#             cropped_coords = (Tracking.Deeplabcut & dict(key, tracking_method=2)).fetch1(
#                 'cropped_x0', 'cropped_x1', 'cropped_y0', 'cropped_y1')
#         else:
#             raise ValueError(
#                 'Fitting is corrupted! Ensure that Tracking.Deeplabcut & dict(key, tracking_method=2) is unique!')

#         self.cropped_coords = cropped_coords

#         # now filter out by std
#         self.std = 5.5
#         fitting_ts = (FittedPupil & key).fetch1('fitting_ts')

#         if fitting_ts < datetime.datetime(2019, 7, 11):

#             # circle
#             center, radius = (FittedPupil.Circle() &
#                               key).fetch('center', 'radius')
#             data = np.vstack((center, radius)).T

#             rejected_inds = DLC_tools.filter_by_std(
#                 data, key['fitting_method'], self.std)

#             center[rejected_inds] = None, None
#             radius[rejected_inds] = None

#             self.circle_center = center
#             self.circle_radius = radius
#             self.circle_rejected_inds = rejected_inds
#             # ellipse':
#             center, major_r, minor_r, rotation_angle = (FittedPupil.Ellipse() & key).fetch(
#                 'center', 'major_radius', 'minor_radius', 'rotation_angle')

#             data = np.vstack((center, major_r, minor_r)).T

#             rejected_inds = DLC_tools.filter_by_std(
#                 data, key['fitting_method'], self.std)

#             center[rejected_inds] = None, None
#             major_r[rejected_inds] = None
#             minor_r[rejected_inds] = None
#             rotation_angle[rejected_inds] = None

#             self.ellipse_center = center
#             self.major_r = major_r
#             self.minor_r = minor_r
#             self.rotation_angle = rotation_angle
#             self.ellipse_rejected_inds = rejected_inds

#         else:
#             # if fit_type == 'circle':
#             center, radius = (FittedPupil.Circle() &
#                               key).fetch('center', 'radius')
#             self.circle_center = center
#             self.circle_radius = radius
#             self.rejected_inds = np.isnan(radius)

#             # elif fit_type == 'ellipse':
#             center, major_r, minor_r, rotation_angle = (FittedPupil.Ellipse() & key).fetch(
#                 'center', 'major_radius', 'minor_radius', 'rotation_angle')
#             self.ellipse_center = center
#             self.major_r = major_r
#             self.minor_r = minor_r
#             self.rotation_angle = rotation_angle
#             self.ellipse_rejected_inds = np.isnan(major_r)

#         if self.key['tracking_method'] == 2:
#             dlc_config = (ConfigDeeplabcut & (
#                 Tracking.Deeplabcut & self.key)).fetch1()

#             config = auxiliaryfunctions.read_config(
#                 dlc_config['config_path'])
#             config['config_path'] = dlc_config['config_path']
#             config['shuffle'] = dlc_config['shuffle']
#             config['trainingsetindex'] = dlc_config['trainingsetindex']
#             config['cropped_coords'] = self.cropped_coords

#             # find path to original video symlink
#             base_path = os.path.splitext(self.avi_path)[0] + '_tracking'
#             video_path = os.path.join(
#                 base_path, os.path.basename(self.avi_path))

#             config['orig_video_path'] = video_path

#             pupil_fit = DLC_tools.DeeplabcutPupilFitting(
#                 config=config, bodyparts='all', cropped=True)

#             # reject outliers
#             pupil_fit.tf_likelihood_array[self.rejected_inds] = False
#             self.dlc_fit = pupil_fit

#     def plot(self, start, end=None, fit_type='circle', ax=None, show_contour=True, mask_flag=True):

#         fit_type = fit_type.lower()
#         if fit_type not in ['circle', 'ellipse']:
#             raise ValueError('fit_type must be either a circle or an ellipse')

#         if end == None:
#             end = start + 1

#         if ax is None:
#             fig = plt.figure(frameon=False, figsize=(8, 6))
#             ax = fig.add_subplot(1, 1, 1)
#             plt.subplots_adjust(left=0, bottom=0, right=1,
#                                 top=1, wspace=0, hspace=0)

#         # check end is less than nframes
#         if end > self.cap.get(cv2.CAP_PROP_FRAME_COUNT):
#             raise ValueError(
#                 'frame num: {} is bigger than the video frame number!'.format(end))

#         plt.xlim(0, self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) -
#                  self.cropped_coords[0])
#         plt.ylim(0, self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) -
#                  self.cropped_coords[2])

#         plt.gca().invert_yaxis()

#         if not show_contour:

#             if fit_type == 'circle':

#                 for frame_num in range(start, end):

#                     ind = frame_num-start

#                     self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
#                     _, img = self.cap.read()
#                     img = np.flip(img, 2)[self.cropped_coords[2]: self.cropped_coords[3], 
#                                         self.cropped_coords[0]: self.cropped_coords[1]]

#                     ax_frame = ax.imshow(img, cmap='gray')

#                     print(frame_num)

#                     plt_fit = plt.Circle((self.circle_center[frame_num][0], 
#                                         self.circle_center[frame_num][1]), 
#                                         self.circle_radius[frame_num], color='b', fill=False)
                    
#                     ax.axis('off')
#                     ax.set_title('frame num: ' + str(frame_num), fontsize=10)
#                     plt.tight_layout()

#                     fig.canvas.draw()

#                     # check if we plot only a single frame. If so, return fig and ax
#                     if end == start + 1:
#                         return fig, ax

#                     display.clear_output(wait=True)
#                     display.display(pl.gcf())
#                     time.sleep(0.5)

#                     plt.cla()
#                 plt.close('all')
            
#             elif fit_type == 'ellipse':
#                 for frame_num in range(start, end):

#                     ind = frame_num-start

#                     self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
#                     _, img = self.cap.read()
#                     img = np.flip(img, 2)[self.cropped_coords[2]: self.cropped_coords[3], 
#                                         self.cropped_coords[0]: self.cropped_coords[1]]

#                     ax_frame = ax.imshow(img, cmap='gray')

#                     plt_fit = Ellipse((self.ellipse_center[ind][0]-self.cropped_coords[0], 
#                                     self.ellipse_center[ind][1]-self.cropped_coords[2]),
#                                     self.major_r[ind]*2, self.minor_r[ind]*2, 
#                                     self.rotation_angle[ind]-90, color='b', fill=False)
                    
#                     ax.axis('off')
#                     ax.set_title('frame num: ' + str(frame_num), fontsize=10)
#                     plt.tight_layout()

#                     fig.canvas.draw()

#                     # check if we plot only a single frame. If so, return fig and ax
#                     if end == start + 1:
#                         return fig, ax

#                     display.clear_output(wait=True)
#                     display.display(pl.gcf())
#                     time.sleep(0.5)

#                     plt.cla()
#                 plt.close('all')

#         if show_contour:

#             # manual tracking
#             if self.key['tracking_method'] == 1:
#                 frame_id, contours = (Tracking.ManualTracking & self.key & 'frame_id >= {}'.format(
#                     start) & 'frame_id < {}'.format(end+1)).fetch('frame_id', 'contour', order_by='frame_id')

#                 for counting_ind, frame_id_ind in enumerate(frame_id):
#                     if self.rejected_inds[frame_id_ind] == False:
#                         contours[counting_ind] = None

#                 if fit_type == 'circle':

#                     for frame_num in range(start, end):

#                         ind = frame_num-start

#                         self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
#                         _, img = self.cap.read()
#                         img = np.flip(img, 2)[self.cropped_coords[2]: self.cropped_coords[3], 
#                                             self.cropped_coords[0]: self.cropped_coords[1]]

#                         ax_frame = ax.imshow(img, cmap='gray')

#                         plt_fit = plt.Circle((self.circle_center[ind][0], 
#                                             self.circle_center[ind][1]), 
#                                             self.circle_radius[ind], color='b', fill=False)

#                         ax.add_patch(plt_fit)
#                         ax_scatter = ax.scatter(contours[ind].squeeze()[:, 0],
#                                                 contours[ind].squeeze()[:, 1]
#                                                 ,
#                                                 s=4**2, color='red', alpha=.5)
                        
#                         ax.axis('off')
#                         ax.set_title('frame num: ' + str(frame_num), fontsize=10)
#                         plt.tight_layout()

#                         fig.canvas.draw()

#                         # check if we plot only a single frame. If so, return fig and ax
#                         if end == start + 1:
#                             return fig, ax

#                         display.clear_output(wait=True)
#                         display.display(pl.gcf())
#                         time.sleep(0.5)

#                         plt.cla()
#                     plt.close('all') 
                
#                 elif fit_type == 'ellipse':
#                     for frame_num in range(start, end):

#                         ind = frame_num-start

#                         self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
#                         _, img = self.cap.read()
#                         img = np.flip(img, 2)[self.cropped_coords[2]: self.cropped_coords[3], 
#                                             self.cropped_coords[0]: self.cropped_coords[1]]

#                         ax_frame = ax.imshow(img, cmap='gray')

#                         plt_fit = Ellipse((self.ellipse_center[ind][0]-self.cropped_coords[0], 
#                                         self.ellipse_center[ind][1]-self.cropped_coords[2]),
#                                         self.major_r[ind]*2, self.minor_r[ind]*2, 
#                                         self.rotation_angle[ind]-90, color='b', fill=False)

#                         ax.add_patch(plt_fit)
#                         ax_scatter = ax.scatter(contours[ind].squeeze()[:, 0] - self.cropped_coords[0],
#                                             contours[ind].squeeze()[:, 1] -
#                                             self.cropped_coords[2],
#                                             s=4**2, color='red', alpha=.5)
#                         ax.axis('off')
#                         ax.set_title('frame num: ' + str(frame_num), fontsize=10)
#                         plt.tight_layout()

#                         fig.canvas.draw()

#                         # check if we plot only a single frame. If so, return fig and ax
#                         if end == start + 1:
#                             return fig, ax

#                         display.clear_output(wait=True)
#                         display.display(pl.gcf())
#                         time.sleep(0.5)

#                         plt.cla()
#                     plt.close('all')

#             # DLC tracking
#             elif self.key['tracking_method'] == 2:

#                 if fit_type == 'circle':

#                     for frame_num in range(start, end):

#                         ind = frame_num-start

#                         self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
#                         _, img = self.cap.read()
#                         img = np.flip(img, 2)[self.cropped_coords[2]: self.cropped_coords[3], 
#                                             self.cropped_coords[0]: self.cropped_coords[1]]

#                         ax_frame = ax.imshow(img, cmap='gray')

#                         plt_fit = plt.Circle((self.circle_center[ind][0], 
#                                             self.circle_center[ind][1]), 
#                                             self.circle_radius[ind], color='b', fill=False)

#                         ax.add_patch(plt_fit)
#                         ax_scatter = ax.scatter(contours[ind].squeeze()[:, 0] ,
#                                                 contours[ind].squeeze()[:, 1] ,
#                                                 s=4**2, color='red', alpha=.5)
                        
#                         ax.axis('off')
#                         ax.set_title('frame num: ' + str(frame_num), fontsize=10)
#                         plt.tight_layout()

#                         fig.canvas.draw()

#                         # check if we plot only a single frame. If so, return fig and ax
#                         if end == start + 1:
#                             return fig, ax

#                         display.clear_output(wait=True)
#                         display.display(pl.gcf())
#                         time.sleep(0.5)

#                         plt.cla()
#                     plt.close('all') 
                
#                 elif fit_type == 'ellipse':
#                     for frame_num in range(start, end):

#                         ind = frame_num-start

#                         self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
#                         _, img = self.cap.read()
#                         img = np.flip(img, 2)[self.cropped_coords[2]: self.cropped_coords[3], 
#                                             self.cropped_coords[0]: self.cropped_coords[1]]

#                         ax_frame = ax.imshow(img, cmap='gray')

#                         plt_fit = Ellipse((self.ellipse_center[ind][0]-self.cropped_coords[0], 
#                                         self.ellipse_center[ind][1]-self.cropped_coords[2]),
#                                         self.major_r[ind]*2, self.minor_r[ind]*2, 
#                                         self.rotation_angle[ind]-90, color='b', fill=False)

#                         ax.add_patch(plt_fit)
#                         ax_scatter = ax.scatter(contours[ind].squeeze()[:, 0] - self.cropped_coords[0],
#                                             contours[ind].squeeze()[:, 1] -
#                                             self.cropped_coords[2],
#                                             s=4**2, color='red', alpha=.5)
#                         ax.axis('off')
#                         ax.set_title('frame num: ' + str(frame_num), fontsize=10)
#                         plt.tight_layout()

#                         fig.canvas.draw()

#                         # check if we plot only a single frame. If so, return fig and ax
#                         if end == start + 1:
#                             return fig, ax

#                         display.clear_output(wait=True)
#                         display.display(pl.gcf())
#                         time.sleep(0.5)

#                         plt.cla()
#                     plt.close('all')

#                 for frame_num in range(start, end):
#                     bpindex, df_x_coords, df_y_coords = self.dlc_fit.coords_pcutoff(frame_num)
                

#             for frame_num in range(start, end):
#                 self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
#                 _, img = self.cap.read()
#                 img = np.flip(img, 2)[
#                     self.cropped_coords[2]: self.cropped_coords[3], self.cropped_coords[0]: self.cropped_coords[1]]

#                 ax_frame = ax.imshow(img, cmap='gray')

#                 if contours[frame_num-start] is not None:
#                     ind = frame_num-start

#                     if fit_type == 'circle':

#                         plt_fit = plt.Circle((center[ind][0]-self.cropped_coords[0], center[ind][1]-self.cropped_coords[2]),
#                                              radius[ind], color='b', fill=False)

#                     else:
#                         plt_fit = Ellipse((center[ind][0]-self.cropped_coords[0], center[ind][1]-self.cropped_coords[2]),
#                                           major_r[ind]*2, minor_r[ind]*2, angle[ind]-90, color='b', fill=False)

#                     ax.add_patch(plt_fit)
#                     ax_scatter = ax.scatter(contours[ind].squeeze()[:, 0] - self.cropped_coords[0],
#                                             contours[ind].squeeze()[:, 1] -
#                                             self.cropped_coords[2],
#                                             s=4**2, color='red', alpha=.5)
#                     # ax.add_collection(ax_scatter)

#                 ax.axis('off')
#                 ax.set_title('frame num: ' + str(frame_num), fontsize=10)
#                 plt.tight_layout()

#                 fig.canvas.draw()

#                 # check if we plot only a single frame. If so, return fig and ax
#                 if end == start + 1:
#                     return fig, ax

#                 display.clear_output(wait=True)
#                 display.display(pl.gcf())
#                 time.sleep(0.5)

#                 plt.cla()
#             plt.close('all')
