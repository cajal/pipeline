import warnings
from pprint import pprint

import datajoint as dj
import pandas as pd

from djaddon import hdf5

try:
    from pupil_tracking import PupilTracker
except ImportError:
    warnings.warn("Failed to import pupil_tacking library. You won't be able to populate trk.EyeFrame")

schema = dj.schema('pipeline_pupiltracking', locals())
from . import rf
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
import glob


@schema
class Roi(dj.Manual):
    definition = """
    # table that stores the correct ROI of the Eye in the video
    ->rf.Eye
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
    pupil_tracker_param_id                      : int            # id for param collection
    ---
    convex_weight_high = Null                 : float        # parameter for tracking
    convex_weight_low = Null                 : float        # parameter for tracking
    thres_perc_high = Null                 : float        # parameter for tracking
    thres_perc_low = Null                 : float        # parameter for tracking
    pupil_left_limit = Null                 : float        # parameter for tracking
    pupil_right_limit = Null                 : float        # parameter for tracking
    min_radius = Null                 : float        # parameter for tracking
    max_radius = Null                   : float       # parameter for tracking
    centre_dislocation_penalty             : float     # parameter for tracking
    distance_sq_pow                         : float       # parameter for tracking

    """

    contents = [
        {'pupil_tracker_param_id': 0, 'convex_weight_high': 0.5,  'convex_weight_low': 0.5,  'thres_perc_high': 99, 'distance_sq_pow': 1,
            'thres_perc_low': 1, 'pupil_left_limit': 0.2, 'pupil_right_limit': 0.8, 'min_radius': 5, 'max_radius': 180,
            'centre_dislocation_penalty': 0.001},
        {'pupil_tracker_param_id': 1, 'convex_weight_high': 0.5, 'convex_weight_low': 0.5, 'thres_perc_high': 98, 'distance_sq_pow': 0.5,
            'thres_perc_low': 2, 'pupil_left_limit': 0.2, 'pupil_right_limit': 0.8, 'min_radius': 5, 'max_radius': 180,
            'centre_dislocation_penalty': 0.05}
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
        print("Populating: ")
        # embed()
        param = (ParamEyeFrame() & 'pupil_tracker_param_id=1').fetch.as_dict()[0]
        # key.update(param)
        key['pupil_tracker_param_id'] = param['pupil_tracker_param_id']
        pprint(key)
        eye_roi = (Roi() & key).fetch1['x_roi_min', 'y_roi_min', 'x_roi_max', 'y_roi_max']
        print("Populating for trk.Roi and roi = ", eye_roi)
        p, f = (rf.Session() & key).fetch1['hd5_path', 'file_base']
        n = (rf.Scan() & key).fetch1['file_num']
        avi_path = glob.glob(r"{p}/{f}{n}*.avi".format(f=f, p=p, n=n))
        # print(avi_path)
        # embed()
        assert len(avi_path) == 1, "Found 0 or more than 1 videos: {videos}".format(videos=str(avi_path))

        tr = PupilTracker(param)
        trace = tr.track_without_svm(avi_path[0], eye_roi)

        # CODE to insert data after tracking
        print("Tracking complete... Now inserting data to datajoint")
        # embed()
        efd = EyeFrame.Detection()
        for index, data in trace.iterrows():
            key['frame'] = index
            self.insert1(key)
            if pd.notnull(data['pupil_x']):
                values = data.to_dict()
                values.update(key)
                efd.insert1(values)

    class Detection(dj.Part):
        definition = """
        # eye frames with detected eye
        ->EyeFrame
        ---
        pupil_x                     : float                         # pupil x position
        pupil_y                     : float                         # pupil y position
        pupil_r_minor               : float                         # pupil radius minor axis
        pupil_r_major               : float                         # pupil radius major axis
        pupil_angle                 : float                         # angle of major axis vs. horizontal axis in radians
        pupil_x_std                 : float                         # pupil x position std
        pupil_y_std                 : float                         # pupil y position std
        pupil_r_minor_std            : float                         # pupil radius minor axis std
        pupil_r_major_std           : float                         # pupil radius major axis std
        pupil_angle_std             : float                         # angle of major axis vs. horizontal axis in radians
        intensity_std               : float                         # standard deviation of the ROI pixel values
        """


@schema
class SelectionProtocol(dj.Lookup):
    definition = """
    # groups of filtering steps to reject bad frames

    filter_protocol_id      : int   # id of the filtering protocol
    ---
    protocol_name           : char(50) # descriptive name of the protocol
    """

    contents = [
        {'filter_protocol_id': 0, 'protocol_name': 'frame_intensity'},
        {'filter_protocol_id': 1, 'protocol_name': 'int_and_ran_pupil_x_50_2'},
        {'filter_protocol_id': 2, 'protocol_name': 'int_and_ran_pupil_x_75_2'},
        {'filter_protocol_id': 3, 'protocol_name': 'int_and_ran_pupil_x_25_2'},
        {'filter_protocol_id': 4, 'protocol_name': 'int_and_ran_pupil_pos'},
        {'filter_protocol_id': 5, 'protocol_name': 'int_and_ran_pupil_pos_spikes_removed'},
        {'filter_protocol_id': 6, 'protocol_name': 'int_and_ran_pupil_pos_spike_filter2'}
    ]

    def apply(self, frames, key):
        print("Applying filter with protocol id :", key['filter_protocol_id'])
        for step in (ProtocolStep() & key).fetch.order_by('priority').as_dict():
            # embed()
            print("....for protocol id:", key['filter_protocol_id'], "applying filter with filter_id = ",
                  step['filter_id'])
            frames = FrameSelector().apply(frames, step, param=step['filter_param'])
        return frames


@schema
class FrameSelector(dj.Lookup):
    definition = """
    # single filters to reject frames
    filter_id           : tinyint   # id of the filter
    ---
    filter_name         : char(50)   # descriptive name of the filter
    """

    contents = [
        {'filter_id': 0, 'filter_name': 'intensity_filter'},
        {'filter_id': 1, 'filter_name': 'ran_pupil_x_th'},
        {'filter_id': 2, 'filter_name': 'ran_pupil_pos'},
        {'filter_id': 3, 'filter_name': 'spike_filter'},
        {'filter_id': 4, 'filter_name': 'spike_filter2'}
    ]

    def apply(self, frames, key, param):
        """
        Apply takes a restriction of EyeFrame.Detection() and returns an even more restricted set of frames

        :param frames: restriction of EyeFrame.Detection()
        :param key: key that singles out a single filter
        :param param: parameters to the filter
        :return: an even more restricted set of frames
        """
        which = (self & key).fetch1['filter_name']

        if which == 'intensity_filter':
            i = frames.fetch['intensity_std']
            th = np.percentile(i, param[0]) / param[1]
            return frames & 'intensity_std>{threshold}'.format(threshold=th)

        if which == 'ran_pupil_x_th':
            i = frames.fetch['pupil_x_std']
            th = np.percentile(i, param[0])
            return frames & 'pupil_x_std<{threshold}*{param}'.format(threshold=th, param=param[1])

        if which == 'ran_pupil_pos':
            i = frames.fetch['pupil_x_std']
            j = frames.fetch['pupil_y_std']
            pos = i*i + j*j
            th = np.percentile(pos, param[0])
            return frames & '(pupil_x_std*pupil_x_std + pupil_y_std*pupil_y_std)<{threshold}*{param}'.format(threshold=th, param=param[1])

        if which == 'spike_filter':
            ra = frames.fetch.order_by('frame')['pupil_r_minor']
            fr = frames.fetch.order_by('frame')['frame']
            slope_coll = []
            for i in range(1,ra.size):
                slope_coll.append((ra[i] - ra[i-1])/ (fr[i] - fr[i-1]))
            slope_coll1 = abs(np.asarray(slope_coll))
            frames_rej = [dict(frame=k) for k in fr[np.where(slope_coll1 > param)]]
            return frames - frames_rej

        if which == 'spike_filter2':
            ra = frames.fetch.order_by('frame')['pupil_r_minor']
            fr = frames.fetch.order_by('frame')['frame']
            fr_rej=[]
            for i in range(2, ra.size-2):
                avg = (ra[i-2] + ra[i-1] + ra[i+1] + ra[i+2]) / 4
                if abs(ra[i] - avg) > param:
                    fr_rej.append(fr[i])
            frames_rej = [dict(frame=k) for k in fr_rej]
            return frames - frames_rej


@schema
class ProtocolStep(dj.Lookup):
    definition = """
    # single filter in a protocol to accept frames
    -> SelectionProtocol
    -> FrameSelector
    priority                : int   # priority of the filter step, the low the higher the priority
    ---
    filter_param=null       : longblob # parameters that are passed to the filter
    """

    # define the protocols. Each protocol has one id, but can have several filters
    contents = [  # parameter needs to be an array
        # protocol 0 contains only one filter and is based on intensity
        {'filter_protocol_id': 0, 'filter_id': 0, 'priority': 50, 'filter_param': np.array([50,2])},
        # protocol 1 = intensity filter + ransac(50,2)
        {'filter_protocol_id': 1, 'filter_id': 0, 'priority': 10, 'filter_param': np.array([50,2])},
        {'filter_protocol_id': 1, 'filter_id': 1, 'priority': 40, 'filter_param': np.array([50,2])},
        # protocol 2 = intensity filter + ransac(75,2)
        {'filter_protocol_id': 2, 'filter_id': 0, 'priority': 10, 'filter_param': np.array([50,2])},
        {'filter_protocol_id': 2, 'filter_id': 1, 'priority': 40, 'filter_param': np.array([75,2])},
        # protocol 3 = intensity filter + ransac(25,2)
        {'filter_protocol_id': 3, 'filter_id': 0, 'priority': 10, 'filter_param': np.array([50,2])},
        {'filter_protocol_id': 3, 'filter_id': 1, 'priority': 40, 'filter_param': np.array([25,2])},
        # protocol 4 = intensity filter + ransac x2+y2
        {'filter_protocol_id': 4, 'filter_id': 0, 'priority': 10, 'filter_param': np.array([50,2])},
        {'filter_protocol_id': 4, 'filter_id': 2, 'priority': 40, 'filter_param': np.array([97,2])},
        # protocol 5 = intensity filter + ransac x2+y2 + spike filter
        {'filter_protocol_id': 5, 'filter_id': 0, 'priority': 10, 'filter_param': np.array([50,2])},
        {'filter_protocol_id': 5, 'filter_id': 2, 'priority': 40, 'filter_param': np.array([97,2])},
        {'filter_protocol_id': 5, 'filter_id': 3, 'priority': 50, 'filter_param': np.array(50)},
        # protocol 6 = intensity filter + ransac x2+y2 + spike filter2
        {'filter_protocol_id': 6, 'filter_id': 0, 'priority': 10, 'filter_param': np.array([50,2])},
        {'filter_protocol_id': 6, 'filter_id': 2, 'priority': 40, 'filter_param': np.array([97,2])},
        {'filter_protocol_id': 6, 'filter_id': 4, 'priority': 50, 'filter_param': np.array(35)}
    ]


@schema
class SelectedFrame(dj.Computed):
    definition = """
    # This schema only contains detected frames that meet a particular quality criterion
    -> EyeFrame.Detection
    -> SelectionProtocol
    ---
    """

    @property
    def populated_from(self):
        return rf.Eye() * SelectionProtocol() & EyeFrame()

    def _make_tuples(self, key):
        print("Key = ", key)
        # embed()
        frames = EyeFrame.Detection() & key
        print('\tLength before filtering: {l}'.format(l=len(frames)))
        # & key can be removed from the line below
        frames = (SelectionProtocol() & key).apply(frames, key)
        print('\tLength after filtering: {l}'.format(l=len(frames)))

        for frame_key in frames.project().fetch.as_dict:
            key.update(frame_key)
            self.insert1(key)


@schema
class Quality(dj.Computed):
    definition = """
    # quality assessment of tracking using Jake's tracked frames as ground truth
    -> rf.Eye
    -> SelectionProtocol
    ---
    pos_err       : float # mean Euclidean distance between pupil positions
    r_corr         : float # correlation of radii
    excess_frames : int   # number of frames detected by tracking but not in Jake's data
    missed_frames : int   # number of frames detected by Jake but no by tracking
    total_frames  : int   # total number of frames in the video
    nan_in_rf     : int   # nan frames in a video in rf.EyeFrame
    """

    @property
    def populated_from(self):
        return rf.Eye().project() * SelectionProtocol() & EyeFrame().project() & rf.EyeFrame().project() & SelectedFrame().project()

    def _make_tuples(self, key):
        # TODO: This function needs cleanup. Only keep relevant stuff for computing the comparisons
        # TODO: Don't plot in _make_tuples. Make plotting an extra function.

        roi_rf = (rf.Eye() & key).fetch['eye_roi']
        # print("Debug 1")
        # embed()
        print("Populating for key= ", key)
        pos_errors = np.zeros(len(rf.EyeFrame() & key))
        r_errors = np.zeros(len(rf.EyeFrame() & key))
        excess_frames = 0
        missed_frames = 0
        r_rf = []
        r_trk = []
        p_err = []
        total_frames = len(rf.EyeFrame() & key)
        miss = []
        indexes = []
        efi = []
        for frame_key in (rf.EyeFrame() & key).project().fetch.as_dict:
            # from IPython import embed
            # print("Debug 2")
            # embed()
            if np.isnan((rf.EyeFrame() & frame_key).fetch['pupil_x']):
                # cant use fetch1 here as it might error out !!
                if (EyeFrame.Detection() & (SelectedFrame() & key) & frame_key).fetch['pupil_x'].shape[0] != 0:
                    excess_frames += 1
                    efi.append(frame_key['frame'])
            else:
                if (EyeFrame.Detection() & frame_key & (SelectedFrame() & key)).fetch['pupil_x'].shape[0] == 0:
                    missed_frames += 1
                    # embed()
                    miss.append(frame_key['frame'])

                else:
                    rx, ry = (rf.EyeFrame() & frame_key).fetch1['pupil_x','pupil_y']
                    tx, ty = (EyeFrame.Detection() & frame_key).fetch1['pupil_x','pupil_y']

                    d_x = rx - tx + roi_rf[0][0][0] - 1
                    d_y = ry - ty + roi_rf[0][0][2] - 1
                    r_rf.append((rf.EyeFrame() & frame_key).fetch1['pupil_r'])
                    r_trk.append((EyeFrame.Detection() & frame_key).fetch1['pupil_r_major'])
                    pos_errors[frame_key['frame']] = pow(d_x, 2) + pow(d_y, 2)
                    indexes.append(frame_key['frame'])
                    p_err.append(pow(d_x, 2) + pow(d_y, 2))
                    if frame_key['frame'] % 1000 is 0:
                        print("Frame Computing = ", frame_key['frame'], " / ", total_frames)
        #embed()
        frames_computed = np.sum(~np.isnan((rf.EyeFrame() & key).fetch['pupil_x'])) - missed_frames
        # frames_computed = len(np.where(np.isnan((rf.EyeFrame() & key).fetch['pupil_x']) == False)[0]) - missed_frames
        key['pos_err'] = pow(np.sum(pos_errors) / frames_computed, 0.5)
        key['r_corr'] = np.corrcoef(r_rf, r_trk)[0][1]
        key['excess_frames'] = excess_frames
        key['missed_frames'] = missed_frames
        key['total_frames'] = total_frames
        key['nan_in_rf'] = np.sum(~np.isnan((rf.EyeFrame() & key).fetch['pupil_x']))
        self.insert1(key)

    def plot_comparison(self):
        #embed()
        # TODO: Make this a proper plotting function
        N = 5
        fig, ax = plt.subplots(1, 2)
        ind = np.arange(N)
        # width = 0.35
        x0 = (self & 'filter_protocol_id=0').fetch['pos_err']
        x1 = (self & 'filter_protocol_id=1').fetch['pos_err']
        x2 = (self & 'filter_protocol_id=2').fetch['pos_err']
        x3 = (self & 'filter_protocol_id=3').fetch['pos_err']
        x4 = (self & 'filter_protocol_id=4').fetch['pos_err']
        # means = [np.mean(x0), np.mean(x1), np.mean(x2), np.mean(x3)]
        # std = [np.std(x0), np.std(x1), np.std(x2), np.std(x3)]

        rects0 = ax[0].bar(0, np.mean(x0), color='r', ecolor='k', align='center', yerr=np.std(x0))
        rects1 = ax[0].bar(1, np.mean(x1), color='b', ecolor='k', align='center', yerr=np.std(x1))
        rects2 = ax[0].bar(2, np.mean(x2), color='g', ecolor='k', align='center', yerr=np.std(x2))
        rects3 = ax[0].bar(3, np.mean(x3), color='y', ecolor='k', align='center', yerr=np.std(x3))
        rects4 = ax[0].bar(4, np.mean(x4), color='m', ecolor='k', align='center', yerr=np.std(x4))
        ax[0].plot(ind, [x0,x1,x2,x3,x4], '-o')
        ax[0].set_xticks(ind)
        label0 = r'$\mu =%.2f\pm\sigma =%.2f$' % (np.mean(x0), np.std(x0))
        label1 = r'$\mu =%.2f\pm\sigma =%.2f$' % (np.mean(x1), np.std(x1))
        label2 = r'$\mu =%.2f\pm\sigma =%.2f$' % (np.mean(x2), np.std(x2))
        label3 = r'$\mu =%.2f\pm\sigma =%.2f$' % (np.mean(x3), np.std(x3))
        label4 = r'$\mu =%.2f\pm\sigma =%.2f$' % (np.mean(x4), np.std(x4))
        lbls = SelectionProtocol().fetch['protocol_name']
        ax[0].set_xticklabels(lbls, rotation=45, ha='right')
        ax[0].set_ylabel('RMSE pupil centre position')
        ax[0].set_xlabel('Filter Protocol ID')
        ax[0].legend((rects0[0], rects1[0], rects2[0], rects3[0], rects4[0]), (label0, label1, label2, label3, label4))

        nan = (self & 'filter_protocol_id=0').fetch['nan_in_rf']
        mf = (self & 'filter_protocol_id=0').fetch['missed_frames']
        ef = (self & 'filter_protocol_id=0').fetch['excess_frames']
        p = (nan-mf)/(nan-mf+ef)
        r = (nan-mf)/nan
        pts0 = ax[1].plot(r, p, 'ok', color='r')

        nan = (self & 'filter_protocol_id=1').fetch['nan_in_rf']
        mf = (self & 'filter_protocol_id=1').fetch['missed_frames']
        ef = (self & 'filter_protocol_id=1').fetch['excess_frames']
        p = (nan-mf)/(nan-mf+ef)
        r = (nan-mf)/nan
        pts1 = ax[1].plot(r, p, 'ok', color='b')

        nan = (self & 'filter_protocol_id=2').fetch['nan_in_rf']
        mf = (self & 'filter_protocol_id=2').fetch['missed_frames']
        ef = (self & 'filter_protocol_id=2').fetch['excess_frames']
        p = (nan-mf)/(nan-mf+ef)
        r = (nan-mf)/nan
        pts2 = ax[1].plot(r, p, 'ok', color='g')

        nan = (self & 'filter_protocol_id=3').fetch['nan_in_rf']
        mf = (self & 'filter_protocol_id=3').fetch['missed_frames']
        ef = (self & 'filter_protocol_id=3').fetch['excess_frames']
        p = (nan-mf)/(nan-mf+ef)
        r = (nan-mf)/nan
        pts3 = ax[1].plot(r, p, 'ok', color='y')

        nan = (self & 'filter_protocol_id=4').fetch['nan_in_rf']
        mf = (self & 'filter_protocol_id=4').fetch['missed_frames']
        ef = (self & 'filter_protocol_id=4').fetch['excess_frames']
        p = (nan-mf)/(nan-mf+ef)
        r = (nan-mf)/nan
        pts4 = ax[1].plot(r, p, 'ok', color='m')

        ax[1].legend((pts0[0], pts1[0], pts2[0], pts3[0], pts4[0]), tuple(lbls), loc=5)
        ax[1].set_ylabel('Precision values')
        ax[1].set_xlabel('Recall values')
        ax[1].set_ylim((0, 1.05))
        ax[1].set_xlim((0, 1.05))
        fig.tight_layout()
        fig.savefig('err_pup_x_with_fil_pr.png')
