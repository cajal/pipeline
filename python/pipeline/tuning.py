"""
Analysis of visual tuning: receptive fields, tuning curves, pixelwise maps
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import convolve

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
    stim_duration      : float                         # (s) total stimulus duration
    """

    key_source = preprocess.Spikes() * preprocess.Sync() & vis.Monet()

    class Map(dj.Part):
        definition = """   #
        -> MonetRF
        -> preprocess.Spikes.RateTrace
        ---
        map : longblob
        """

        def save(self, path="."):
            """
            save RF maps into PNG files
            """
            import os
            from matplotlib.image import imsave
            from matplotlib import pyplot as plt
            cmap = plt.get_cmap('seismic')
            for key in self.fetch.keys():
                map = (MonetRF.Map() & key).fetch1['map']
                scale = np.abs(map).max()*2
                for i in range(map.shape[2]):
                    filename = os.path.join(
                        os.path.expanduser(path),
                        '{animal_id:05d}_{session}_scan{scan_idx:02d}_method{extract_method}'
                        '-{spike_method}_{trace_id:03d}-{frame}.png'.format(frame=i, **key))
                    print(filename)
                    imsave(filename, cmap(map[:, :, i]/scale+0.5))

    def _make_tuples(self, key):

        def hamming(half, dim):
            """ normalized hamming kernel """
            k = np.hamming(np.floor(half)*2+1)
            return k.reshape([1]*dim+[k.size])/k.sum()

        # enter basic information about the RF Map
        nbins = 6
        bin_size = 0.1     # s
        [x, y, distance, diagonal] = (preprocess.Sync() * vis.Session() & key).fetch1[
            'resolution_x', 'resolution_y', 'monitor_distance', 'monitor_size']
        cm_per_inch = 2.54
        degrees_per_pixel = 180 / np.pi * diagonal * cm_per_inch / np.sqrt(np.float64(x)*x + np.float64(y)*y) / distance
        degrees_x = degrees_per_pixel * x
        degrees_y = degrees_per_pixel * y

        # fetch traces and their slices (for galvo scans)
        trace_time = (preprocess.Sync() & key).fetch1['frame_times'].squeeze()  # calcium scan frame times
        traces, slices, trace_keys = (
            preprocess.Spikes.RateTrace() * preprocess.ExtractRaw.GalvoROI() &
            key).fetch['rate_trace', 'slice', dj.key]
        n_slices = (preprocess.Prepare.Galvo() & key).fetch1['nslices']
        trace_time = trace_time[:n_slices*traces[0].size]  # truncate if interrupted scan
        assert n_slices*traces[0].size == trace_time.size, 'trace times must be a multiple of n_slices'
        dt = (trace_time[1:]-trace_time[:-1]).mean()*n_slices
        maps = [0]*len(traces)
        n_trials = 0
        stim_duration = 0.0
        for trial_key in (preprocess.Sync() * vis.Trial() * vis.Condition() &
                          'trial_idx between first_trial and last_trial' &
                          vis.Monet() & key).fetch.order_by('trial_idx').keys():
            print('Trial', trial_key['trial_idx'], flush=True, end=' - ')
            n_trials += 1

            # get the movie and scale to [-1, +1]
            movie_times = (vis.Trial() & trial_key).fetch1['flip_times'].flatten()
            stim_duration += movie_times[-1] - movie_times[0]
            movie = (vis.Monet() * vis.MonetLookup() & trial_key).fetch1['cached_movie']
            movie = (np.float32(movie) - 127.5) / 126.5

            # rebin the movie to bin_size.  Reverse time for convoluion.
            fps = 1 / (movie_times[1:] - movie_times[:-1]).mean()
            start_time = movie_times[0] + bin_size / 2
            movie = interp1d(
                movie_times, convolve(
                    movie, hamming(bin_size*fps, 2), 'same'))(np.r_[movie_times[-1]:start_time:-bin_size])

            # compute the maps for each slice separately to account for time differences between slices
            for islice in set(slices):
                print('Slice', islice, end=' ', flush=True)
                ix = np.where(slices == islice)[0]
                time = trace_time[islice-1::n_slices]

                # rebin traces to bin_size
                snippets = interp1d(time, convolve(
                    np.stack(traces[ix].flatten()),
                    hamming(bin_size/dt, 1), 'same'), fill_value=0)(
                    np.r_[start_time+bin_size*(nbins-1):movie_times[-1]:bin_size])

                # correlate snippents to movie
                for i, snippet in zip(ix, snippets):
                    maps[i] += convolve(
                        movie,
                        snippet.reshape((1, 1, snippet.size))/np.linalg.norm(snippet),
                        mode='valid')
            print()

        # submit data
        self.insert1(dict(key,
                          degrees_x=degrees_x,
                          degrees_y=degrees_y,
                          nbins=nbins,
                          bin_size=bin_size * 1000,
                          stim_duration=stim_duration))
        MonetRF.Map().insert(
            (dict(trace_key, map=m/n_trials) for trace_key, m in zip(trace_keys, maps)),
            ignore_extra_fields=True)
        print('Done')


@schema
class DirectionalResponse(dj.Computed):
    definition = """  # response to directional stimulus
    -> Directional
    -> preprocess.Spikes
    ---
    latency : float   # latency used (ms)
    """

    class TraceTrial(dj.Part):
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
            preprocess.Spikes.RateTrace() * preprocess.ExtractRaw.GalvoROI() &
            key).fetch['rate_trace', 'slice', dj.key]
        traces = np.float64(np.stack(t.flatten() for t in traces))

        #  fetch and clean up the trace time
        trace_time = (preprocess.Sync() & key).fetch1['frame_times'].squeeze()  # calcium scan frame times
        n_slices = (preprocess.Prepare.Galvo() & key).fetch1['nslices']
        trace_time = trace_time[:n_slices*traces.shape[1]]  # truncate if interrupted scan
        assert n_slices*traces[0].size == trace_time.size, 'trace times must be a multiple of n_slices'
        trace_time = trace_time[::n_slices]    # keep trace times for the first slice only
        dt = (trace_time[1:]-trace_time[:-1]).mean()

        # compute and interpolate cumulative traces on time of first slice
        assert traces.ndim == 2 and traces.shape[0] == len(trace_keys), 'incorrect trace dimensions'
        traces = interp1d(trace_time, np.cumsum(traces, axis=1)*dt)

        # insert responses for each trace and trial with time adjustment for slices
        latency = 0.01  # s
        self.insert1(dict(key, latency=1000*latency))
        table = DirectionalResponse.TraceTrial()
        for onset, offset, trial_key in zip(*(Directional.Trial() & key).fetch['onset', 'offset', dj.key]):
            for islice in set(slices):
                ix = np.where(slices == islice)[0]
                responses = (traces(offset+latency-dt*islice) -
                             traces(onset+latency-dt*islice))[ix]/(offset-onset)
                table.insert((dict(trial_key, response=response, **trace_keys[i])
                              for i, response in zip(ix, responses)),
                             ignore_extra_fields=True)
        print('Done')


schema.spawn_missing_classes()
