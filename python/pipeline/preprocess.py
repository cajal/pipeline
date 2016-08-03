import datajoint as dj
from . import experiment, vis, PipelineException
from warnings import warn
import numpy as np
import sh
import os
from scipy import integrate as integr
from .utils.dsp import mirrconv
from .utils.eye_tracking import ROIGrabber, ts2sec, read_video_hdf5, PupilTracker
from scipy import signal
from scipy import stats
from . import config
from distutils.version import StrictVersion

assert StrictVersion(dj.__version__) >= StrictVersion('0.2.8')

schema = dj.schema('pipeline_preprocess', locals())


def notnan(x, start=0, increment=1):
    while np.isnan(x[start]) and start < len(x) and start >= 0:
        start += increment
    return start


def normalize(img):
    return (img - img.min()) / (img.max() - img.min())


def erd():
    """a shortcut for convenience"""
    dj.ERD(schema).draw(prefix=False)


@schema
class Slice(dj.Lookup):
    definition = """  # slices in resonant scanner scans
    slice  : tinyint  # slice in scan
    """
    contents = ((i,) for i in range(12))


@schema
class Channel(dj.Lookup):
    definition = """  # recording channel, directly related to experiment.PMTFilterSet.Channel
    channel : tinyint
    """
    contents = [[1], [2], [3], [4]]


@schema
class Prepare(dj.Imported):
    definition = """  # master table that gathers data about the scans of different types, prepares for trace extraction
    -> experiment.Scan
    """

    class Galvo(dj.Part):
        definition = """    # basic information about resonant microscope scans, raster correction
        -> Prepare
        ---
        nframes_requested       : int               # number of valumes (from header)
        nframes                 : int               # frames recorded
        px_width                : smallint          # pixels per line
        px_height               : smallint          # lines per frame
        um_width                : float             # width in microns
        um_height               : float             # height in microns
        bidirectional           : tinyint           # 1=bidirectional scanning
        fps                     : float             # (Hz) frames per second
        zoom                    : decimal(4,1)      # zoom factor
        dwell_time              : float             # (us) microseconds per pixel per frame
        nchannels               : tinyint           # number of recorded channels
        nslices                 : tinyint           # number of slices
        slice_pitch             : float             # (um) distance between slices
        fill_fraction           : float             # raster scan fill fraction (see scanimage)
        preview_frame           : longblob          # raw average frame from channel 1 from an early fragment of the movie
        raster_phase            : float             # shift of odd vs even raster lines
        """

    class GalvoMotion(dj.Part):
        definition = """   # motion correction for galvo scans
        -> Prepare.Galvo
        -> Slice
        ---
        -> Channel
        template                    : longblob       # stack that was used as alignment template
        motion_xy                   : longblob       # (pixels) y,x motion correction offsets
        motion_rms                  : float          # (um) stdev of motion
        align_times=CURRENT_TIMESTAMP: timestamp     # automatic
        """

    class GalvoAverageFrame(dj.Part):
        definition = """   # average frame for each slice and channel after corrections
        -> Prepare.GalvoMotion
        -> Channel
        ---
        frame  : longblob     # average frame ater Anscombe, max-weighting,
        """

    class Aod(dj.Part):
        definition = """   # information about AOD scans
        -> Prepare
        """

    class AodPoint(dj.Part):
        definition = """  # points in 3D space in coordinates of an AOD scan
        -> Prepare.Aod
        point_id : smallint    # id of a scan point
        ---
        x: float   # (um)
        y: float   # (um)
        z: float   # (um)
        """


@schema
class Method(dj.Lookup):
    definition = """  #  methods for extraction from raw data for either AOD or Galvo data
    extract_method :  tinyint
    """
    contents = [[1], [2], [3], [4]]

    class Aod(dj.Part):
        definition = """
        -> Method
        ---
        description  : varchar(60)
        high_pass_stop=null : float   # (Hz)
        low_pass_stop=null  : float   # (Hz)
        subtracted_princ_comps :  tinyint  # number of principal components to subtract
        """
        contents = [
            [1, 'raw traces', None, None, 0],
            [2, 'band pass, -1 princ comp', 0.02, 20, -1],
        ]

    class Galvo(dj.Part):
        definition = """  # extraction methods for galvo
        -> Method
        ---
        segmentation  :  varchar(16)   #
        """
        contents = [
            [1, 'manual'],
            [2, 'nmf']
        ]


@schema
class ExtractRaw(dj.Imported):
    definition = """  # pre-processing of a twp-photon scan
    -> Prepare
    -> Method
    """

    @property
    def key_source(self):
        return Prepare() * Method().proj() & dj.OrList(
            Prepare.Aod() * Method.Aod(), Prepare.Galvo() * Method.Galvo())

    class Trace(dj.Part):
        definition = """  # raw trace, common to Galvo
        -> ExtractRaw
        -> Channel
        trace_id  : smallint
        ---
        raw_trace : longblob     # unprocessed calcium trace
        """

    class GalvoSegmentation(dj.Part):
        definition = """  # segmentation of galvo movies
        -> ExtractRaw
        -> Slice
        ---
        segmentation_mask=null  :  longblob
        """

    class GalvoROI(dj.Part):
        definition = """  # region of interest produced by segmentation
        -> ExtractRaw.GalvoSegmentation
        -> ExtractRaw.Trace
        ---
        mask_pixels          :longblob      # indices into the image in column major (Fortran) order
        mask_weights = null  :longblob      # weights of the mask at the indices above
        """

        @staticmethod
        def reshape_masks(mask_pixels, mask_weights, px_height, px_width):
            ret = np.zeros((px_height, px_width, len(mask_pixels)))
            for i, (mp, mw) in enumerate(zip(mask_pixels, mask_weights)):
                mask = np.zeros(px_height * px_width)
                mask[mp.squeeze().astype(int) - 1] = mw.squeeze()
                ret[..., i] = mask.reshape(px_height, px_width, order='F')
            return ret

    class SpikeRate(dj.Part):
        definition = """
        # spike trace extracted while segmentation
        -> ExtractRaw.Trace
        ---
        spike_trace :longblob
        """

    def plot_galvo_ROIs(self, outdir='./'):
        import seaborn as sns
        import matplotlib.pyplot as plt

        sns.set_context('paper')
        theCM = sns.blend_palette(['lime', 'gold', 'deeppink'], n_colors=10)  # plt.cm.RdBu_r
        # theCM = plt.cm.get_cmap('viridis')

        for key in (self * self.GalvoSegmentation().proj() * Method.Galvo() & dict(segmentation='nmf')).fetch.as_dict:
            mask_px, mask_w, spikes, traces = \
                (self.GalvoROI() * \
                 self.SpikeRate() * ComputeTraces().Trace() & key & \
                 dict(segmentation=2)).fetch.order_by('trace_id')['mask_pixels', 'mask_weights', 'spike_trace', 'trace']

            template = np.stack([normalize(t)
                                 for t in (Prepare.GalvoAverageFrame() & key).fetch['frame']], axis=2).max(axis=2)

            d1, d2, fps = tuple(map(int, (Prepare.Galvo() & key).fetch1['px_height', 'px_width', 'fps']))
            hs = int(np.round(fps * 60))
            masks = self.GalvoROI.reshape_masks(mask_px, mask_w, d1, d2)
            try:
                sh.mkdir('-p', os.path.expanduser(outdir) + '/scan_idx{scan_idx}/slice{slice}'.format(**key))
            except:
                pass
            gs = plt.GridSpec(6, 2)

            N = len(spikes)
            for cell, (sp_trace, ca_trace) in enumerate(zip(spikes, traces)):
                print(
                    "{cell:03d}/{N}: animal_id {animal_id}\tsession {session}\tscan_idx {scan_idx:02d}\t{segmentation}\tslice {slice}".format(
                        cell=cell + 1, N=N, **key))
                sp_trace = sp_trace.squeeze()
                ca_trace = ca_trace.squeeze()
                with sns.axes_style('white'):
                    fig = plt.figure(figsize=(6, 8))
                    ax_image = fig.add_subplot(gs[:-2, 0])

                with sns.axes_style('ticks'):
                    ax_small_tr = fig.add_subplot(gs[1, 1])
                    ax_small_ca = fig.add_subplot(gs[2, 1], sharex=ax_small_tr)
                    ax_sp = fig.add_subplot(gs[-1, :])
                    ax_tr = fig.add_subplot(gs[-2, :], sharex=ax_sp)

                # --- plot zoom in
                n = len(sp_trace)
                tmp = np.array(sp_trace)
                tmp[np.isnan(tmp)] = 0
                loc = np.argmax(np.convolve(tmp, np.ones(hs) / hs, mode='same'))
                loc = max(loc - hs // 2, 0)
                loc = n - hs if loc > n - hs else loc

                ax_small_tr.plot(sp_trace[loc:loc + hs], 'k', lw=1)
                ax_small_ca.plot(ca_trace[loc:loc + hs], 'k', lw=1)

                # --- plot traces
                ax_sp.plot(sp_trace, 'k', lw=1)

                ax_sp.fill_between([loc, loc + hs], np.zeros(2), np.ones(2) * np.nanmax(sp_trace),
                                   color='steelblue', alpha=0.5)
                ax_tr.plot(ca_trace, 'k', lw=1)
                ax_tr.fill_between([loc, loc + hs], np.zeros(2), np.ones(2) * np.nanmax(ca_trace),
                                   color='steelblue', alpha=0.5)
                ax_image.imshow(template, cmap=plt.cm.gray)
                ax_image.contour(masks[..., cell], colors=theCM, zorder=10)

                fig.suptitle(
                    "animal_id {animal_id}:session {session}:scan_idx {scan_idx}:{segmentation}:slice{slice}:cell{cell}".format(
                        cell=cell + 1, **key))

                sns.despine(fig)
                ax_sp.set_title('NMF spike trace', fontweight='bold')
                ax_tr.set_title('Raw trace', fontweight='bold')
                ax_small_tr.set_title('NMF spike trace', fontweight='bold')
                ax_small_ca.set_title('Raw trace', fontweight='bold')
                ax_sp.axis('tight')
                for a in [ax_small_ca, ax_small_tr, ax_sp, ax_tr]:
                    a.set_xticks([])
                    a.set_yticks([])
                    sns.despine(ax=a, left=True)
                ax_sp.set_xlabel('time')
                fig.tight_layout()
                plt.savefig(
                    outdir + "/scan_idx{scan_idx}/slice{slice}/cell{cell:03d}_animal_id_{animal_id}_session_{session}.png".format(
                        cell=cell + 1, **key))
                plt.close(fig)

    def _make_tuples(self, key):
        """ implemented in matlab """
        raise NotImplementedError("Implemented in Matlab")


@schema
class Sync(dj.Imported):
    definition = """
    -> Prepare
    ---
    -> vis.Session
    first_trial                 : int                           # first trial index from vis.Trial overlapping recording
    last_trial                  : int                           # last trial index from vis.Trial overlapping recording
    signal_start_time           : double                        # (s) signal start time on stimulus clock
    signal_duration             : double                        # (s) signal duration on stimulus time
    frame_times = null          : longblob                      # times of frames and slices
    sync_ts=CURRENT_TIMESTAMP   : timestamp                     # automatic
    """


@schema
class ComputeTraces(dj.Computed):
    definition = """   # compute traces
    -> ExtractRaw
    ---
    """

    class Trace(dj.Part):
        definition = """  # final calcium trace but before spike extraction or filtering
        -> ComputeTraces
        trace_id             : smallint                     #
        ---
        trace = null         : longblob                     # leave null same as ExtractRaw.Trace
        """

    @property
    def key_source(self):
        return (ExtractRaw() & ExtractRaw.Trace()).proj()

    @staticmethod
    def get_band_emission(fluorophore, center, band_width):
        pass_band = (center - band_width / 2, center + band_width / 2)
        nu_loaded, s_loaded = (experiment.Fluorophore.EmissionSpectrum()
                               & dict(fluorophore=fluorophore, loaded=1)).fetch1['wavelength', 'fluorescence']

        nu_free, s_free = (experiment.Fluorophore.EmissionSpectrum()
                           & dict(fluorophore=fluorophore, loaded=0)).fetch1['wavelength', 'fluorescence']

        f_loaded = lambda xx: np.interp(xx, nu_loaded, s_loaded)
        f_free = lambda xx: np.interp(xx, nu_free, s_free)
        return integr.quad(f_free, *pass_band)[0], integr.quad(f_loaded, *pass_band)[0]

    @staticmethod
    def estimate_twitch_ratio(x, y, fps, df1, df2):
        # low pass filter for unsharp masking
        hh = signal.hamming(2 * np.round(fps / 0.03) + 1)
        hh /= hh.sum()

        # high pass filter for heavy denoising
        hl = signal.hamming(2 * np.round(fps / 8) + 1)
        hl /= hl.sum()
        x = mirrconv(x - mirrconv(x, hh), hl)
        y = mirrconv(y - mirrconv(y, hh), hl)

        slope, intercept, _, p, _ = stats.linregress(x, y)
        slope = -1 if slope >= 0 else slope
        return df2 / df1 / slope

    def _make_tuples(self, key):
        if ExtractRaw.Trace() & key:
            fluorophore = (experiment.Session.Fluorophore() & key).fetch1['fluorophore']
            if fluorophore != 'Twitch2B':
                print('Populating', key)

                def remove_channel(x):
                    x.pop('channel')
                    return x

                self.insert1(key)
                self.Trace().insert(
                    [remove_channel(x) for x in (ExtractRaw.Trace() & key).proj(trace='raw_trace').fetch.as_dict])
            elif fluorophore == 'Twitch2B':
                # --- get channel indices and filter passbands for twitch settings
                filters = experiment.PMTFilterSet() * experiment.PMTFilterSet.Channel() \
                          & dict(pmt_filter_set='2P3 blue-green A')
                fps = (Prepare.Galvo() & key).fetch1['fps']
                green_idx, green_center, green_pb = \
                    (filters & dict(color='green')).fetch1['pmt_channel', 'spectrum_center', 'spectrum_bandwidth']
                blue_idx, blue_center, blue_pb = \
                    (filters & dict(color='blue')).fetch1['pmt_channel', 'spectrum_center', 'spectrum_bandwidth']

                # --- compute theoretical emission over filter spectra
                g_free, g_loaded = self.get_band_emission(fluorophore, green_center, green_pb)
                b_free, b_loaded = self.get_band_emission(fluorophore, blue_center, blue_pb)
                dg = g_loaded - g_free
                db = b_loaded - b_free

                green = (ExtractRaw.Trace() & dict(key, channel=green_idx)).proj(green='channel',
                                                                                 green_trace='raw_trace')
                blue = (ExtractRaw.Trace() & dict(key, channel=blue_idx)).proj(blue='channel',
                                                                               blue_trace='raw_trace')

                self.insert1(key)
                for trace_id, gt, bt in zip(*(green * blue).fetch['trace_id', 'green_trace', 'blue_trace']):
                    print(
                        '\tProcessing animal_id: {animal_id}\t session: {session}\t scan_idx: {scan_idx}\ttrace: {trace_id}'.format(
                            trace_id=trace_id, **key))
                    gt, bt = gt.squeeze(), bt.squeeze()
                    start = notnan(gt * bt)
                    end = notnan(gt * bt, len(gt) - 1, increment=-1)
                    gamma = self.estimate_twitch_ratio(gt[start:end], bt[start:end], fps, dg, db)

                    x = np.zeros_like(gt) * np.NaN
                    gt, bt = gt[start:end], bt[start:end]
                    r = (gt - bt) / (gt + bt)
                    x[start:end] = (-b_free + g_free * gamma - r * (b_free + g_free * gamma)) / \
                                   (db - dg * gamma + r * (db + dg * gamma))

                    trace_key = dict(key, trace_id=trace_id, trace=x.astype(np.float32)[:, None])
                    self.Trace().insert1(trace_key)


@schema
class SpikeMethod(dj.Lookup):
    definition = """
    spike_method   :  smallint   # spike inference method
    ---
    spike_method_name     : varchar(16)   #  short name to identify the spike inference method
    spike_method_details  : varchar(255)  #  more details about
    language :  enum('matlab','python')   #  implementation language
    """

    contents = [
        [2, "fastoopsi", "nonnegative sparse deconvolution from Vogelstein (2010)", "matlab"],
        [3, "stm", "spike triggered mixture model from Theis et al. (2016)", "python"],
        [4, "improved oopsi", "", "matlab"],
        [5, "nmf", "", "matlab"]
    ]

    def spike_traces(self, X, fps):
        try:
            import c2s
        except:
            warn("c2s was not found. You won't be able to populate ExtracSpikes")
        assert self.fetch1['language'] == 'python', "This tuple cannot be computed in python."
        if self.fetch1['spike_method'] == 3:
            N = len(X)
            for i, trace in enumerate(X):
                print('Predicting trace %i/%i' % (i + 1, N))
                tr0 = np.array(trace.pop('trace').squeeze())
                start = notnan(tr0)
                end = notnan(tr0, len(tr0) - 1, increment=-1)
                trace['calcium'] = np.atleast_2d(tr0[start:end + 1])

                trace['fps'] = fps
                data = c2s.preprocess([trace], fps=fps)
                data = c2s.predict(data, verbosity=0)

                tr0[start:end + 1] = data[0].pop('predictions')
                data[0]['rate_trace'] = tr0.T
                data[0].pop('calcium')
                data[0].pop('fps')

                yield data[0]


@schema
class Spikes(dj.Computed):
    definition = """  # infer spikes from calcium traces
    -> ComputeTraces
    -> SpikeMethod
    """

    @property
    def key_source(self):
        return (ComputeTraces() * SpikeMethod() & [dict(spike_method_name='stm'), dict(spike_method_name='nmf')]).proj()
        # return (ComputeTraces() * SpikeMethod() & dict(spike_method_name='nmf')).proj()

    class RateTrace(dj.Part):
        definition = """  # Inferred
        -> Spikes
        -> ExtractRaw
        trace_id  : smallint
        ---
        rate_trace = null  : longblob     # leave null same as ExtractRaw.Trace
        """

    def plot_traces(self, outdir='./'):

        import matplotlib.pyplot as plt
        import seaborn as sns
        gs = plt.GridSpec(2, 5)
        for key in (ComputeTraces.Trace() & self).fetch.keys():
            print('Processing', key)
            fps = (Prepare.Galvo() & key).fetch1['fps']

            hs = int(np.round(fps * 30))

            fig = plt.figure(figsize=(10, 4))
            ax_ca = fig.add_subplot(gs[0, :3])
            ax_sp = fig.add_subplot(gs[1, :3], sharex=ax_ca)

            ax_cas = fig.add_subplot(gs[0, 3:], sharey=ax_ca)
            ax_sps = fig.add_subplot(gs[1, 3:], sharex=ax_cas, sharey=ax_sp)

            ca = (ComputeTraces.Trace() & key).fetch1['trace'].squeeze()
            t = np.arange(len(ca)) / fps
            ax_ca.plot(t, ca, 'k')
            loc = None
            for sp, meth in zip(*(self.RateTrace() * SpikeMethod() & key).fetch['rate_trace', 'spike_method_name']):
                ax_sp.plot(t, sp, label=meth)
                # --- plot zoom in
                if loc is None:
                    n = len(sp)
                    tmp = np.array(sp)
                    tmp[np.isnan(tmp)] = 0
                    loc = np.argmax(np.convolve(tmp, np.ones(hs) / hs, mode='same'))
                    loc = max(loc - hs // 2, 0)
                    loc = n - hs if loc > n - hs else loc
                    ax_cas.plot(t[loc:loc + hs], ca[loc:loc + hs], 'k')
                    ax_ca.fill_between([t[loc], t[loc + hs - 1]], np.nanmin(ca) * np.ones(2),
                                       np.nanmax(ca) * np.ones(2),
                                       color='dodgerblue', zorder=-10)
                ax_sps.plot(t[loc:loc + hs], sp[loc:loc + hs], label=meth)

            ax_sp.set_xlabel('time [s]')
            ax_sps.set_xlabel('time [s]')

            ax_sp.legend()
            ax_sps.legend()

            try:
                sh.mkdir('-p', os.path.expanduser(outdir) + '/session{session}/scan_idx{scan_idx}/'.format(**key))
            except:
                pass

            fig.tight_layout()
            plt.savefig(outdir \
                        + "/session{session}/scan_idx{scan_idx}/trace{trace_id:03d}_animal_id_{animal_id}.png".format(
                **key))
            plt.close(fig)

    def _make_tuples(self, key):
        print('Populating', key)
        if (SpikeMethod() & key).fetch1['spike_method_name'] == 'stm':
            prep = (Prepare() * Prepare.Aod() & key) or (Prepare() * Prepare.Galvo() & key)
            fps = prep.fetch1['fps']
            X = (ComputeTraces.Trace() & key).proj('trace').fetch.as_dict()
            self.insert1(key)
            for x in (SpikeMethod() & key).spike_traces(X, fps):
                self.RateTrace().insert1(dict(key, **x))
        elif (SpikeMethod() & key).fetch1['spike_method_name'] == 'nmf':
            if ExtractRaw.SpikeRate() & key:
                self.insert1(key)
                for x in (ExtractRaw.SpikeRate() & key).fetch.as_dict:
                    x['rate_trace'] = x.pop('spike_trace')
                    x.pop('channel')
                    self.RateTrace().insert1(dict(key, **x))
        else:
            raise NotImplementedError('Method {spike_method} not implemented.'.format(**key))

@schema
class EyeQuality(dj.Lookup):
    definition = """
    # Different eye quality definitions for Tracking
    
    eye_quality                : smallint
    ---
    description                : varchar(255)
    """

    contents = [
        (-1, 'unusable'),
        (0,  'good quality'),
        (1,  'poor quality'),
    ]

@schema
class Eye(dj.Imported):
    definition = """
    # eye velocity and timestamps

    -> experiment.Scan
    ---
    -> EyeQuality
    eye_roi                     : tinyblob  # manual roi containing eye in full-size movie
    eye_time                    : longblob  # timestamps of each frame in seconds, with same t=0 as patch and ball data
    total_frames                : int       # total number of frames in movie.
    eye_ts=CURRENT_TIMESTAMP    : timestamp # automatic
    """

    def unpopulated(self):
        """
        Returns all keys from Scan()*Session() that are not in Eye but have a video.


        :param path_prefix: prefix to the path to find the video (usually '/mnt/', but empty by default)
        """

        rel = experiment.Session() * experiment.Scan.EyeVideo()
        path_prefix = config['path.mounts']
        restr = [k for k in (rel - self).proj('behavior_path', 'filename').fetch.as_dict() if
                 os.path.exists("{path_prefix}/{behavior_path}/{filename}".format(path_prefix=path_prefix, **k))]
        return (rel - self) & restr

    def grab_timestamps_and_frames(self, key, n_sample_frames=100):

        import cv2
        path_prefix = config['path.mounts']

        rel = experiment.Session() * experiment.Scan.EyeVideo() * experiment.Scan.WheelFile().proj(hdf_file='filename')

        info = (rel & key).fetch1()
        avi_path = "{path_prefix}/{behavior_path}/{filename}".format(path_prefix=path_prefix, **info)

        # replace number by %d for hdf-file reader
        tmp = info['hdf_file'].split('.')
        info['hdf_file'] = tmp[0][:-1] + '%d.' + tmp[-1]

        hdf_path = "{path_prefix}/{behavior_path}/{hdf_file}".format(path_prefix=path_prefix, **info)

        data = read_video_hdf5(hdf_path)
        packet_length = data['analogPacketLen']
        dat_time, _, _ = ts2sec(data['ts'], packet_length)

        cam_key = 'cam1ts' if info['rig'] == '2P3' else  'cam2ts'
        eye_time, _, _ = ts2sec(data[cam_key], packet_length)
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
        key['eye_time'], frames, key['total_frames'] = self.grab_timestamps_and_frames(key)
        rg = ROIGrabber(frames.mean(axis=2))
        print(EyeQuality())
        key['eye_quality'] = int(input("Enter the quality of the eye: ")) 
        key['eye_roi'] = rg.roi
        self.insert1(key)
        print('[Done]')

@schema
class TrackingParameters(dj.Lookup):
    definition = """
    # table that stores the paths for the params for pupil_tracker

    -> EyeQuality
    ---
    thres_perc_high              : float        # parameter for tracking
    thres_perc_low               : float        # parameter for tracking
    convex_weight_low            : float        # threshold will be the convex combination of the low and high percentile weighted
    relative_area_threshold      : float        # enclosing rotating rectangle has to have at least that amount of area
    ratio_threshold              : float        # ratio of major and minor radius cannot be larger than this
    error_threshold              : float        # threshold on the RMSE of the ellipse fit
    min_contour_len              : int          # minimal required contour length (must be at least 5)
    margin                       : float        # relative margin the pupil center should not be in
    """

    contents = [
        {'eye_quality': 0,
         'thres_perc_high': 98,
         'thres_perc_low': 10,
         'convex_weight_low': 0.5,
         'relative_area_threshold': 0.01,
         'ratio_threshold': 1.5,
         'error_threshold': 0.15,
         'min_contour_len': 5,
         'margin': 0.2
         },
        {'eye_quality': 1,
         'thres_perc_high': 98,
         'thres_perc_low': 10,
         'convex_weight_low': 0.3,
         'relative_area_threshold': 0.01,
         'ratio_threshold': 1.3,
         'error_threshold': 0.2,
         'min_contour_len': 5,
         'margin': 0.2
         },
    ]

@schema
class EyeTracking(dj.Computed):
    definition = """
    -> Eye
    -> TrackingParameters
    ---
    tracking_ts=CURRENT_TIMESTAMP    : timestamp  # automatic
    """

    class Frame(dj.Part):
        definition = """
        -> EyeTracking
        frame_id            : int           # frame id with matlab based 1 indexing
        ---
        rotated_rect=NULL        : tinyblob      # rotated rect (center, sidelength, angle) containing the ellipse
        contour=NULL             : longblob      # eye contour relative to ROI
        center=NULL              : tinyblob      # center of the ellipse in (x, y) of image
        major_r=NULL             : float         # major radius of the ellipse
        frame_intensity=NULL     : float         # std of the frame
        """
    @property
    def key_source(self):
        return (Eye()*TrackingParameters()).proj()

    def _make_tuples(self, key):
        print("Populating", key)
        param = (TrackingParameters() & key).fetch1()

        roi = (Eye() & key).fetch1['eye_roi']

        video_info = (experiment.Session() * experiment.Scan.EyeVideo() & key).fetch1()
        avi_path = "{path_prefix}/{behavior_path}/{filename}".format(path_prefix=config['path.mounts'], **video_info)

        tr = PupilTracker(param)
        traces = tr.track(avi_path, roi-1) # -1 because of matlab indices

        self.insert1(key)
        fr = self.Frame()
        for trace in traces:
            trace.update(key)
            fr.insert1(trace)



schema.spawn_missing_classes()
