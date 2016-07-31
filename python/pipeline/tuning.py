"""
Analysis of visual tuning: receptive fields, tuning curves, pixelwise maps
"""

import numpy as np
import datajoint as dj
from . import preprocess, vis   # needed for foreign keys

from distutils.version import StrictVersion
assert StrictVersion(dj.__version__) >= StrictVersion('0.2.8')

schema = dj.schema('pipeline_tuning', locals())


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
class OriMap(dj.Imported):
    definition = """ # pixelwise responses to full-field directional stimuli
    -> OriDesignMatrix
    -> preprocess.Prepare.GalvoMotion
    ---
    regr_coef_maps: longblob  # regression coefficients, width x height x nConds
    r2_map: longblob  # pixelwise r-squared after gaussinization
    dof_map: longblob  # degrees of in original signal, width x height
    """


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
class MonetRF(dj.Computed):
    definition = """  # spike-triggered average of receptive fields
    -> preprocess.Sync
    -> preprocess.Spikes
    ---
    nbins              : smallint                      # temporal bins
    bin_size           : float                         # (ms) temporal bin size
    degrees_x          : float                         # degrees along x
    degrees_y          : float                         # degrees along y
    """

    key_source = preprocess.Spikes() * preprocess.Sync() & vis.Monet()

    class Map(dj.Part):
        definition = """   #
        -> MonetRF
        -> preprocess.Spikes.RateTrace
        ---
        map : longblob
        """

    def _make_tuples(self, key):

        def hamming(half, dim):
            k = np.hamming(np.floor(half)*2+1)
            return k.reshape([1]*dim+[k.size])/k.sum()

        from scipy.interpolate import interp1d
        from scipy.signal import convolve

        # enter basic information about the RF Map
        nbins = 6
        bin_size = 0.1     # s
        [x, y, distance, diagonal] = (preprocess.Sync() * vis.Session() & key).fetch1[
            'resolution_x', 'resolution_y', 'monitor_distance', 'monitor_size']
        cm_per_inch = 2.54
        degrees_per_pixel = 180 / np.pi * diagonal * cm_per_inch / np.sqrt(np.float64(x)*x + np.float64(y)*y) / distance
        degrees_x = degrees_per_pixel * x
        degrees_y = degrees_per_pixel * y
        self.insert1(dict(key, degrees_x=degrees_x, degrees_y=degrees_y, nbins=nbins, bin_size=bin_size*1000))

        # fetch traces and their slices (for galvo scans)
        trace_time = (preprocess.Sync() & key).fetch1['frame_times'].squeeze()  # calcium scan frame times
        trace_list, slices, trace_keys = (
            preprocess.Spikes.RateTrace() * preprocess.ExtractRaw.GalvoROI() &
            key).fetch['rate_trace', 'slice', dj.key]
        n_slices = (preprocess.Prepare.Galvo() & key).fetch1['nslices']
        assert n_slices*trace_list[0].size == trace_time.size, 'trace times must be a multiple of n_slices'
        dt = (trace_time[1:]-trace_time[:-1]).mean()/n_slices
        maps = [0]*len(trace_list)
        n_trials = 0
        for trial_key in (preprocess.Sync() * vis.Trial() * vis.Condition() &
                          'trial_idx between first_trial and last_trial' &
                          vis.Monet() & key).fetch.order_by('trial_idx').keys():
            print('Trial', trial_key['trial_idx'], flush=True, end=' - ')
            n_trials += 1
            movie_times = (vis.Trial() & trial_key).fetch1['flip_times'].flatten()
            movie = (vis.Monet() * vis.MonetLookup() & trial_key).fetch1['cached_movie']
            movie = (np.float32(movie) - 127.5) / 126.5
            fps = 1/(movie_times[1:] - movie_times[:-1]).mean()
            start_time = movie_times[0] + bin_size / 2
            movie = interp1d(
                movie_times, convolve(
                    movie, hamming(bin_size*fps, 2), 'same'))(np.r_[start_time:movie_times[-1]:bin_size])
            for islice in set(slices):
                print('Slice', islice, end=' ', flush=True)
                ix = np.where(slices == islice)[0]
                time = trace_time[islice-1::n_slices]
                snippets = interp1d(time, convolve(
                    np.stack(trace_list[ix]),
                    hamming(bin_size/dt, 1), 'same'), fill_value=0)(
                    np.r_[start_time+bin_size*(nbins-1):movie_times[-1]:bin_size])
                for i, snippet in zip(ix, snippets):
                    maps[i] += convolve(movie, snippet[::-1].reshape((1, 1, snippet.size)), mode='valid')
            print()
        MonetRF.Map().insert(
            (dict(trace_key, map=m/n_trials) for trace_key, m in zip(trace_keys, maps)),
            ignore_extra_fields=True)
        print('Done')

schema.spawn_missing_classes()
