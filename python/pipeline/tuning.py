"""
Analysis of visual tuning: receptive fields, tuning curves, pixelwise maps
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import convolve
from scipy import linalg, stats
import io
import imageio
import datajoint as dj
from . import preprocess
from . import experiment
from . import vis

from distutils.version import StrictVersion
assert StrictVersion(dj.__version__) >= StrictVersion('0.3.8')


schema = dj.schema('pipeline_tuning', locals())


def hamming(half, dim):
    """ normalized hamming kernel """
    k = np.hamming(np.floor(half) * 2 + 1)
    return k.reshape([1] * dim + [k.size]) / k.sum()


@schema
class CaKernel(dj.Lookup):
    definition = """  # options for calcium response kinetics.
    kernel  : tinyint    # calcium option number
    -----
    transient_shape  : enum('exp','onAlpha')  # calcium transient shape
    latency = 0      : float                  # (s) assumed neural response latency
    tau              : float                  # (s) time constant (used by some integration functions
    explanation      : varchar(255)           # explanation of calcium response kinents
    """

    contents = [
        [0, 'exp', 0.03,  0.5, 'instantaneous rise, exponential delay'],
        [1, 'onAlpha', 0.03, 0.5, 'alpha function response to on onset only'],
        [2, 'exp', 0.03,  1.0, 'instantaneous rise, exponential delay'],
        [3, 'onAlpha', 0.03, 1.0, 'alpha function response to on onset only']
    ]


@schema
class Directional(dj.Computed):
    definition = """  # all directional drift trials for the scan
    -> preprocess.Sync
    ---
    ndirections     : tinyint    # number of directions
    """

    class Trial(dj.Part):
        definition = """ #  directional drift trials
        -> Directional
        drift_trial     : smallint               # trial index
        ---
        -> vis.Trial
        direction                   : float                         # (degrees) direction of drift
        onset                       : double                        # (s) onset time in rf.Sync times
        offset                      : double                        # (s) offset time in rf.Sync times
        """


@schema
class OriDesignMatrix(dj.Computed):
    definition = """  # design matrix for directional response
    -> Directional
    -> CaKernel
    -----
    design_matrix   : longblob   # times x nConds
    regressor_cov   : longblob   # regressor covariance matrix,  nConds x nConds
    """


@schema
class OriMapy(dj.Imported):
    definition = """ # pixelwise responses to full-field directional stimuli
    -> OriDesignMatrix
    -> preprocess.Prepare.GalvoMotion
    ---
    regr_coef_maps: longblob  # regression coefficients, widtlh x height x nConds
    r2_map: longblob  # pixelwise r-squared after gaussinization
    dof_map: longblob  # degrees of in original signal, width x height
    """

    def _make_tuples(self, key):
        # Load the scan
        import scanreader
        scan_filename = (experiment.Scan() & key).local_filenames_as_wildcard
        scan = scanreader.read_scan(scan_filename)
        scan = (scan[key['slice']-1, :, :, 0, :]).astype(np.float32, copy=False)

        # Correct the scan
        correct_motion = (preprocess.Prepare.GalvoMotion() & key).get_correct_motion()
        correct_raster = (preprocess.Prepare.Galvo() & key).get_correct_raster()
        scan = correct_motion(correct_raster(scan))
        design, cov = (OriDesignMatrix() & key).fetch1['design_matrix', 'regressor_cov']
        height, width, nslices = (preprocess.Prepare.Galvo() & key).fetch1('px_height', 'px_width', 'nslices')
        design = design[key['slice'] - 1::nslices, :]
        if scan.shape[2] == 2*design.shape[0]:
            scan = (scan[:,:,::2] + scan[:,:,1::2])/2  # this is a hack for mesoscope scanner -- needs fixing

        assert design.shape[0] == scan.shape[2]
        height, width = scan.shape[0:2]    # hack for mesoscope -- needs fixing
        assert (height, width) == scan.shape[0:2]

        # remove periods where the design matrix has any nans
        ix = np.logical_not(np.isnan(design).any(axis=1))
        design = design[ix, :]
        design = design - design.mean()
        nregressors = design.shape[1]

        # normalize scan
        m = scan.mean(axis=-1, keepdims=True)
        scan -= m
        scan /= m
        v = (scan**2).sum(axis=-1)

        # estimate degrees of freedom per pixel
        spectrum = np.abs(np.fft.fft(scan, axis=-1))
        dof = (spectrum.sum(axis=-1)**2/(spectrum**2).sum(axis=-1)).astype(np.int32)

        # solve
        scan = scan[:, :, ix].reshape((-1, design.shape[0])).T
        x, r2, rank, sv = linalg.lstsq(design, scan, overwrite_a=True, overwrite_b=True, check_finite=False)
        del scan, design

        assert rank == nregressors
        x = x.T.reshape((height, width, -1))
        r2 = 1-r2.reshape((height, width))/v

        self.insert1(dict(key, regr_coef_maps=x, r2_map=r2, dof_map=dof))



@schema
class Cos2Map(dj.Computed):
    definition = """  # pixelwise cosine fit to directional response
    -> OriMap
    -----
    cos2_amp   : longblob   # dF/F at preferred direction
    cos2_r2    : longblob   # fraction of variance explained (after gaussinization)
    cos2_fp    : longblob   # p-value of F-test (after gaussinization)
    pref_ori   : longblob   # (radians) preferred direction
    """


@schema
class DirectionalResponse(dj.Computed):
    definition = """  # response to directional stimulus
    -> Directional
    -> preprocess.Spikes
    ---
    latency : float   # latency used (ms)
    """

    class Trial(dj.Part):
        definition = """   # the response for each trial and each trace
        -> DirectionalResponse
        -> preprocess.Spikes.RateTrace
        -> Directional.Trial
        ---
        response : float   #  integrated response
        """

    def _make_tuples(self, key):
        print('Directional response for ', key)
        traces, slices, trace_keys = (
            preprocess.Spikes.RateTrace() * preprocess.Slice() & preprocess.ExtractRaw.GalvoROI() &
            key).fetch('rate_trace', 'slice', dj.key)
        traces = np.float64(np.stack(t.flatten() for t in traces))

        #  fetch and clean up the trace time
        trace_time = (preprocess.Sync() & key).fetch1('frame_times').squeeze()  # calcium scan frame times
        n_slices = (preprocess.Prepare.Galvo() & key).fetch1('nslices')
        trace_time = trace_time[:n_slices*traces.shape[1]]  # truncate if interrupted scan
        assert n_slices*traces[0].size == trace_time.size, 'trace times must be a multiple of n_slices'
        slice_interval = (trace_time[1:]-trace_time[:-1]).mean()
        frame_interval = slice_interval * n_slices
        trace_time = trace_time[::n_slices]    # keep trace times for the first slice only

        # compute and interpolate cumulative traces on time of first slice
        assert traces.ndim == 2 and traces.shape[0] == len(trace_keys), 'incorrect trace dimensions'
        traces = interp1d(trace_time, np.cumsum(traces, axis=1)*frame_interval)

        # insert responses for each trace and trial with time adjustment for slices
        latency = 0.01  # s
        self.insert1(dict(key, latency=1000*latency))
        table = DirectionalResponse.Trial()
        for onset, offset, trial_key in zip(*(Directional.Trial() & key).fetch('onset', 'offset', dj.key)):
            for islice in set(slices):
                ix = np.where(slices == islice)[0]
                try:
                    responses = (traces(offset+latency-slice_interval*islice) -
                                 traces(onset+latency-slice_interval*islice))[ix]/(offset-onset)
                except ValueError:
                    pass
                else:
                    table.insert((dict(trial_key, response=response, **trace_keys[i])
                                  for i, response in zip(ix, responses)),
                                 ignore_extra_fields=True)
        print('Done')


@schema
class RFMethod(dj.Lookup):
    definition = """
    rf_method :  tinyint    #   rf computation method
    ----
    stim_selection :  varchar(64)   #  stim to use.  If no stimulus, the parts will be missing.
    algorithm      :  varchar(30)   #  short name for the computational approach for computing RF
    """
    contents = [
        (1, 'monet', 'sta'),
        (2, 'monet', 'backprop'),
        (3, 'clips', 'backprop')]


@schema
class RF(dj.Computed):
    definition = """  # spike-triggered average of receptive fields
    -> preprocess.Sync
    -> preprocess.Spikes
    -> RFMethod
    ---
    nbins                : smallint                     # temporal bins
    bin_size             : float                        # (ms) temporal bin size
    degrees_x            : float                        # degrees along x
    degrees_y            : float                        # degrees along y
    stim_duration        : float                        # (s) total stimulus duration
    """

    @property
    def key_source(self):
        return preprocess.Spikes() * RFMethod() & (
            preprocess.Sync() * vis.Trial() & 'trial_idx between first_trial and last_trial')

    class Map(dj.Part):
        definition = """   # spatiotemporal receptive field map
        -> RF
        -> preprocess.Spikes.RateTrace
        ---
        scale : float       # receptive field scale
        map   : longblob    # int8 data map scaled by scale
        """

        def save(self, path="."):
            """
            save RF maps into PNG files
            """
            import os
            from matplotlib.image import imsave
            from matplotlib import pyplot as plt
            cmap = plt.get_cmap('seismic')
            crange = 20.0
            for key in self.fetch.keys():
                data, scale = (RF.Map() & key).fetch1('map', 'scale')
                data = np.float64(data) * scale / 127
                p = os.path.join(
                    os.path.expanduser(path), '{animal_id:05d}_{session}_scan{scan_idx:02d}'.format(**key))
                if not os.path.exists(p):
                    os.makedirs(p)
                for frame in range(1, 2):  # range(data.shape[2]):
                    filename = os.path.join(
                        p, '{trace_id:03d}-meth{extract_method}'
                           '-{spike_method}-{rf_method}_{frame}.png'.format(frame=frame, **key))
                    print(filename)
                    imsave(filename, cmap(data[:, :, frame] / crange + 0.5))

    @staticmethod
    def spike_triggered_avg(snippets, movie, nbins):
        """
        spatiotemporal spike-triggered-average
        :param snippets: traces ndarray of shape (n, t) where n is the trace number and t is time index
        :param movie: ndarray of shape (y, x, t)
        :param nbins: number of temporal bins in the STA.
        :return: ndarray of shape (n, y, x, nbins)
        """
        return np.stack((np.tensordot(snippets, movie[:, :, rf_bin:rf_bin + snippets.shape[1]], axes=(1, 2))
                         for rf_bin in reversed(range(nbins))), 3)

    @staticmethod
    def predict_traces(movie, maps):
        """
        :param movie: ndarray of shape (y, x, t)
        :param maps: ndarray of shape (n, y, x, nbins)
        :return: traces ndarray of shape (n, t)
        """
        nbins = maps.shape[-1]
        return sum(
            np.tensordot(maps[:, :, :, nbins - tau - 1], movie, axes=((1, 2), (0, 1)))[:,
            tau:movie.shape[2] + tau - nbins + 1]
            for tau in range(nbins))

    @staticmethod
    def soft_thresh(maps, lam, mu):
        return (1 - mu) * np.sign(maps) * np.maximum(0, abs(maps) - lam * np.sqrt(
            (maps ** 2).mean(axis=(1, 2, 3), keepdims=True)))

    def _make_tuples(self, key):
        # enter basic information about the RF Map
        print('Populating', key)
        nbins = 5
        bin_size = 0.1  # s
        x, y, distance, diagonal = (preprocess.Sync() * vis.Session() & key).fetch1(
            'resolution_x', 'resolution_y', 'monitor_distance', 'monitor_size')
        cm_per_inch = 2.54
        degrees_per_pixel = 180 / np.pi * diagonal * cm_per_inch / np.sqrt(
            np.float64(x) * x + np.float64(y) * y) / distance
        degrees_x = degrees_per_pixel * x
        degrees_y = degrees_per_pixel * y

        # enumerate all the stimulus types:
        stim_selection, algorithm = (RFMethod() & key).fetch1('stim_selection', 'algorithm')
        try:
            condition_table, condition_identifier = dict(
                monet=lambda: (vis.Monet(), 'rng_seed'),
                clips=lambda: (vis.MovieClipCond(), 'clip_number'))[stim_selection]()
        except KeyError:
            raise NotImplementedError('Unknown stimulus selection')

        trials = (preprocess.Sync() * vis.Trial() * dj.U('cond_idx') * condition_table
                  & 'trial_idx between first_trial and last_trial' & key)
        number_of_repeats = dict(
            dj.U(condition_identifier).aggregate(trials, n='count(*)').fetch())
        trial_keys = list(trials.fetch.order_by('trial_idx').keys())

        if not trial_keys:
            # submit data when no trials are found
            self.insert1(dict(key,
                              degrees_x=degrees_x,
                              degrees_y=degrees_y,
                              nbins=nbins,
                              bin_size=bin_size * 1000,
                              stim_duration=0))
            return

        # fetch traces and their slices (for galvo scans)
        print('fetching traces...', flush=True)
        trace_time = (preprocess.Sync() & key).fetch1['frame_times'].squeeze()  # calcium scan frame times
        traces, trace_keys = (
            preprocess.Spikes.RateTrace() & key).fetch['rate_trace', dj.key]
        n_slices = (preprocess.Prepare.Galvo() & key).fetch1['nslices']
        trace_time = trace_time[:n_slices * traces[0].size]  # truncate if interrupted scan
        assert n_slices * traces[0].size == trace_time.size, 'trace times must be a multiple of n_slices'
        frame_interval = (trace_time[1:] - trace_time[:-1]).mean() * n_slices

        # interpolate traces on times of the first slice, smoothed for bin_size
        n_traces = traces.size
        traces = interp1d(trace_time[::n_slices], convolve(
            np.stack(trace.flatten() for trace in traces),
            hamming(bin_size / frame_interval, 1), 'same'))

        # cache traces
        print('computing STA...', flush=True)
        stim_duration = 0
        cache = {}
        trace_norm = np.zeros(n_traces)
        maps = 0  # spike-triggered average
        for trial_key in trial_keys:
            # load the movies
            print('%d' % trial_key['trial_idx'], flush=True, end=' ')
            movie_times = (vis.Trial() & trial_key).fetch1['flip_times'].flatten()
            fps = 1 / np.diff(movie_times).mean()
            stim_duration += movie_times[-1] - movie_times[0]
            if stim_selection == 'monet':
                movie, cond = (vis.Monet() * vis.MonetLookup() & trial_key).fetch1['cached_movie', 'rng_seed']
                movie = (np.float32(movie) - 127.5) / 126.5  # rescale to [-1, +1]
            elif stim_selection == 'clips':
                movie, cond = (vis.MovieClipCond() * vis.Movie.Clip() & trial_key).fetch1['clip', 'clip_number']
                movie = imageio.get_reader(io.BytesIO(movie.tobytes()), 'ffmpeg')
                movie = np.stack((np.float64(frame).mean(axis=2) * 2 / 255 - 1
                                  for t, frame in zip(movie_times, movie.iter_data())), axis=2)
                # high-pass filter above 1 Hz
                movie -= convolve(movie, hamming(fps, 2), 'same')
            else:
                raise NotImplementedError('invalid stimulus selection')
            # rebin the movie to bin_size.  Reverse time for convolution.
            start_time = movie_times[0] + bin_size / 2
            movie = convolve(movie, hamming(bin_size * fps, 2), 'same')
            movie = interp1d(movie_times, movie)
            movie = movie(np.r_[start_time:movie_times[-1]:bin_size]) / np.sqrt(number_of_repeats[cond])
            if algorithm != 'sta':
                # cache the movie if it's needed for iterations
                cache[trial_key['trial_idx']] = movie
            snippets = traces(np.r_[start_time + bin_size * (nbins - 1):movie_times[-1]:bin_size])
            trace_norm += ((snippets - snippets.mean(axis=1, keepdims=True)) ** 2
                           / number_of_repeats[cond]).sum(axis=1)
            maps += RF.spike_triggered_avg(snippets, movie, nbins)
        del traces
        del snippets

        if algorithm == 'backprop':
            sta = maps
            iterations = 15
            beta = 0.4
            maps = beta * RF.soft_thresh(sta, lam=0.5, mu=0.05)
            print()
            for iteration in range(iterations):
                predicted_sta = 0
                for trial_key in trial_keys:
                    print(end='.', flush=True)
                    movie = cache[trial_key['trial_idx']]
                    predicted_traces = RF.predict_traces(movie, maps)
                    predicted_sta += RF.spike_triggered_avg(predicted_traces, movie, nbins)
                predicted_sta /= np.maximum(1, np.sqrt(
                    (predicted_sta ** 2).sum(axis=(1, 2, 3), keepdims=True) /
                    (sta ** 2).sum(axis=(1, 2, 3), keepdims=True)))
                maps = RF.soft_thresh(maps + beta * (sta - predicted_sta), lam=beta * 0.5, mu=beta * 0.05)
                print('iteration', iteration, flush=True)

        # submit data
        self.insert1(dict(key,
                          degrees_x=degrees_x,
                          degrees_y=degrees_y,
                          nbins=nbins,
                          bin_size=bin_size * 1000,
                          stim_duration=stim_duration))
        RF.Map().insert((dict(trace_key,
                              map=np.int8(127 * m / np.max(abs(m))),
                              scale=np.max(abs(m)) / np.sqrt(n),  **key)
                         for trace_key, m, n in zip(trace_keys, maps, trace_norm)),
                        ignore_extra_fields=True)
        print('done.', flush=True)




schema.spawn_missing_classes()


