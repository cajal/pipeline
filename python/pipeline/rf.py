"""
Computation of receptive fields.
This module pipeline.rf module is part of the same schema as pipeline.tuning and will be merged after debugging.
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import convolve
import io
import imageio
import datajoint as dj
from . import preprocess, vis   # needed for foreign keys


schema = dj.schema('pipeline_tuning', locals())


def hamming(half, dim):
    """ normalized hamming kernel """
    k = np.hamming(np.floor(half) * 2 + 1)
    return k.reshape([1] * dim + [k.size]) / k.sum()


def MatchedTrials():
    """
    Common queries for trials corresponding to a scan
    :return: query corresponding to a scan.
    """
    return (preprocess.Sync() * vis.Trial() * dj.U('cond_idx') &
            'trial_idx between first_trial and last_trial').proj('flip_times')


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
                data, scale = (RF.Map() & key).fetch1['map', 'scale']
                data = np.float64(data)*scale/127
                for frame in range(data.shape[2]):
                    filename = os.path.join(
                        os.path.expanduser(path),
                        '{animal_id:05d}_{session}_scan{scan_idx:02d}_{trace_id:03d}-meth{extract_method}'
                        '-{spike_method}-{rf_method}_{frame}.png'.format(frame=frame, **key))
                    print(filename)
                    imsave(filename, cmap(data[:, :, frame]/crange+0.5))

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
            np.tensordot(maps[:, :, :, nbins-tau-1], movie, axes=((1, 2), (0, 1)))[:, tau:movie.shape[2]+tau-nbins+1]
            for tau in range(nbins))

    @staticmethod
    def soft_thresh(maps, lam, mu):
        return (1-mu)*np.sign(maps)*np.maximum(0, abs(maps)-lam*np.sqrt((maps**2).mean(axis=(1, 2, 3), keepdims=True)))

    def _make_tuples(self, key):
        # enter basic information about the RF Map
        print('Populating', key)
        nbins = 5
        bin_size = 0.1     # s
        [x, y, distance, diagonal] = (preprocess.Sync() * vis.Session() & key).fetch1[
            'resolution_x', 'resolution_y', 'monitor_distance', 'monitor_size']
        cm_per_inch = 2.54
        degrees_per_pixel = 180 / np.pi * diagonal * cm_per_inch / np.sqrt(np.float64(x)*x + np.float64(y)*y) / distance
        degrees_x = degrees_per_pixel * x
        degrees_y = degrees_per_pixel * y

        # enumerate all the stimulus types:
        stim_selection, algorithm = (RFMethod() & key).fetch1['stim_selection', 'algorithm']
        try:
            condition_table, condition_identifier = dict(
                monet=lambda: (vis.Monet(), 'rng_seed'),
                clips=lambda: (vis.MovieClipCond(), 'clip_number'))[stim_selection]()
        except KeyError:
            raise NotImplementedError('Unknown stimulus selection')

        trials = (MatchedTrials() & key) * condition_table
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
        trace_time = trace_time[:n_slices*traces[0].size]  # truncate if interrupted scan
        assert n_slices*traces[0].size == trace_time.size, 'trace times must be a multiple of n_slices'
        frame_interval = (trace_time[1:]-trace_time[:-1]).mean()*n_slices

        # interpolate traces on times of the first slice, smoothed for bin_size
        n_traces = traces.size
        traces = interp1d(trace_time[::n_slices], convolve(
                            np.stack(trace.flatten() for trace in traces),
                            hamming(bin_size/frame_interval, 1), 'same'))

        # cache traces
        print('computing STA...', flush=True)
        stim_duration = 0
        cache = {}
        trace_norm = np.zeros(n_traces)
        maps = 0     # spike-triggered average
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
                movie -= convolve(movie, hamming(fps, 3), 'same')
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
            maps = beta*RF.soft_thresh(sta, lam=0.5, mu=0.05)
            print()
            for iteration in range(iterations):
                predicted_sta = 0
                for trial_key in trial_keys:
                    print(end='.', flush=True)
                    movie = cache[trial_key['trial_idx']]
                    predicted_traces = RF.predict_traces(movie, maps)
                    predicted_sta += RF.spike_triggered_avg(predicted_traces, movie, nbins)
                predicted_sta /= np.maximum(1, np.sqrt(
                    (predicted_sta**2).sum(axis=(1, 2, 3), keepdims=True) /
                    (sta**2).sum(axis=(1, 2, 3), keepdims=True)))
                maps = RF.soft_thresh(maps + beta*(sta - predicted_sta), lam=beta*0.5, mu=beta*0.05)
                print('iteration', iteration, flush=True)

        # submit data        print('\ninserting...', end=' ', flush=True)
        self.insert1(dict(key,
                          degrees_x=degrees_x,
                          degrees_y=degrees_y,
                          nbins=nbins,
                          bin_size=bin_size * 1000,
                          stim_duration=stim_duration))
        RF.Map().insert((dict(trace_key, **key,
                              map=np.int8(127*m/np.max(abs(m))),
                              scale=np.max(abs(m))/np.sqrt(n))
                         for trace_key, m, n in zip(trace_keys, maps, trace_norm)),
                        ignore_extra_fields=True)
        print('done.', flush=True)


