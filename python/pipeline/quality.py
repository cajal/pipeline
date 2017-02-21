from itertools import count

import datajoint as dj
from scipy.interpolate import InterpolatedUnivariateSpline
from .preprocess import Sync, fill_nans, SpikeMethod, Prepare, BehaviorSync, ExtractRaw, ManualSegment, Method
from .preprocess import Spikes as PreSpikes
from . import preprocess, vis
import numpy as np
import pandas as pd
from scipy import integrate

schema = dj.schema('pipeline_quality', locals())


def integration_factory(frame_times, trace):
    def responses(bins):
        if not hasattr(responses, 'spline'):
            responses.spline = InterpolatedUnivariateSpline(frame_times, trace, k=1, ext=1)
        spline = responses.spline
        ret = np.zeros(len(bins) - 1)
        for j, (a, b) in enumerate(zip(bins[:-1], bins[1:])):
            ret[j] = integrate.quad(spline, a, b)[0]

        return ret

    return responses


@schema
class Spikes(dj.Computed):
    definition = """
    -> preprocess.Spikes
    ---
    leading_nans      : bool        # whether or not any of the traces has leading nans
    trailing_nans     : bool        # whether or not any of the traces has trailing nans
    stimulus_nans     : bool        # whether or not any of the traces has nans during the stimulus
    nan_idx=null      : longblob    # boolean array indicating where the nans are
    stimulus_start    : int         # start of the stimulus in matlab 1 based indices
    stimulus_end      : int         # end of the stimulus in matlab 1 based indices
    """

    @property
    def key_source(self):
        return preprocess.Spikes() & preprocess.Sync()

    def _make_tuples(self, key):
        print('Populating', key)
        spikes = np.vstack([s.squeeze() for s in (preprocess.Spikes.RateTrace() & key).fetch['rate_trace']])
        s = spikes.sum(axis=0)
        nans = np.isnan(s)

        key['leading_nans'] = int(nans[0])
        key['trailing_nans'] = int(nans[1])

        t = (preprocess.Sync() & key).fetch1['frame_times']  # does not need to be unique

        flip_first = (vis.Trial() * preprocess.Sync().proj('psy_id', trial_idx='first_trial') & key).fetch1[
            'flip_times']
        flip_last = (vis.Trial() * preprocess.Sync().proj('psy_id', trial_idx='last_trial') & key).fetch1['flip_times']

        # (vis.Trial() * preprocess.Sync() & 'trial_idx between first_trial and last_trial')
        fro = np.atleast_1d(flip_first.squeeze())[0]
        to = np.atleast_1d(flip_last.squeeze())[
            -1]  # not necessarily where the stimulus stopped, just last presentation
        idx_fro = np.argmin(np.abs(t - fro))
        idx_to = np.argmin(np.abs(t - to)) + 1
        key['stimulus_nans'] = int(np.any(nans[idx_fro:idx_to]))
        if np.any(nans):
            key['nan_idx'] = nans
        key['stimulus_start'] = idx_fro + 1
        key['stimulus_end'] = idx_to

        self.insert1(key)


@schema
class IntegrationWindow(dj.Lookup):
    definition = """
    # start & end times in s, and number bins to integrate the neural response over

    integr_id       :  varchar(64) # unique id
    ---
    start           :  float    # start time relative to onset in s
    end             :  float    # end time relative to onset in s
    """

    contents = [
        dict(integr_id='mad_antolik_onset', start=0.05, end=0.350),
        dict(integr_id='mad_antolik_offset', start=0.55, end=0.850),
        dict(integr_id='mad_antolik_baseline', start=1.05, end=1.35)
    ]

    def get_bin_borders(self):
        """
        Returns a numpy array with bin borders. Relation must be restricted to a
        single entry before calling this function.

        Returns: bin borders as numpy array

        """
        return np.array(self.fetch1['start', 'end'])

    def get_bin_centers(self):
        """
        Returns a numpy array with bin centers. Relation must be restricted to a
        single entry before calling this function.

        Returns: bin centers as numpy array

        """
        s, e = self.fetch1['start', 'end']
        return 0.5 * (s + e)


@schema
class SpikeMethods(dj.Computed):
    definition = """
    -> preprocess.ComputeTraces
    spike_method_1  -> preprocess.Spikes
    spike_method_2  -> preprocess.Spikes
    ---

    """

    class Correlation(dj.Part):
        definition = """
        -> SpikeMethods
        -> preprocess.ComputeTraces.Trace
        ---
        corr=null  : float # correlation between spike method 1 and 2 on that trace
        """

    @property
    def key_source(self):
        return preprocess.Spikes().proj(spike_method_1='spike_method') \
               * preprocess.Spikes().proj(spike_method_2='spike_method') \
               & 'spike_method_1<spike_method_2'

    def _make_tuples(self, key):
        tr1 = pd.DataFrame((preprocess.Spikes.RateTrace() & key & dict(spike_method=key['spike_method_1'])).fetch())
        tr2 = pd.DataFrame((preprocess.Spikes.RateTrace() & key & dict(spike_method=key['spike_method_2'])).fetch())
        tr1['rate_trace'] = [np.asarray(e).squeeze() for e in tr1.rate_trace]
        tr2['rate_trace'] = [np.asarray(e).squeeze() for e in tr2.rate_trace]
        trs = tr1.merge(tr2, on=['animal_id', 'session', 'scan_idx', 'extract_method', 'trace_id'], how='inner',
                        suffixes=('_1', '_2'))

        self.insert1(key)
        print('Populating', key)
        for i, row in trs.iterrows():
            trace1, trace2 = tuple(map(np.asarray, (row.rate_trace_1, row.rate_trace_2)))
            idx = ~np.isnan(trace1 + trace2)
            k = row.to_dict()
            k['corr'] = np.corrcoef(trace1[idx], trace2[idx])[0, 1]
            self.Correlation().insert1(k, ignore_extra_fields=True)

@schema
class IntegratedResponse(dj.Computed):
    definition = """
    # responses of one neuron to MadAntolik stimulus

    -> PreSpikes.RateTrace
    -> Sync
    -> IntegrationWindow
    ---
    """

    class Trial(dj.Part):
        definition = """
        -> IntegratedResponse
        -> vis.Trial
        -> vis.MovieStillCond
        -> vis.Movie.Still
        ---
        response           : longblob   # reponse of one neurons for all bins
        behavior_bins      : longblob   # integration borders on the behavior clock
        """

    @property
    def key_source(self):
        return PreSpikes.RateTrace() * Sync() * IntegrationWindow() & 'extract_method in (1,2)'

    @staticmethod
    def _get_spike_trace(key):
        trace0 = (PreSpikes.RateTrace() * SpikeMethod() & key).fetch1['rate_trace'].astype(np.float64)
        return fill_nans(trace0).squeeze()

    @staticmethod
    def _slice(key):
        seg = (Method.Galvo() & key).fetch1['segmentation']
        #------ TODO remove when done -----------
        from IPython import embed
        embed()
        # exit()
        #----------------------------------------
        if seg == 'manual':
            return (ManualSegment() & key).fetch1['slice']
        elif seg == 'nmf':
            return (ExtractRaw.GalvoROI() & key).fetch1['slice']

    @staticmethod
    def _get_stimulus_timiming(key):
        # --- get timing information
        trials = Sync() * vis.Trial() * vis.MovieStillCond() * vis.Movie.Still() \
                 & key & 'trial_idx between first_trial and last_trial'
        frame_times, nslices = (Sync() * Prepare.Galvo() & key).fetch1['frame_times', 'nslices']

        sli = IntegratedResponse._slice(key)
        frame_times = frame_times.squeeze()[sli - 1::nslices]
        flip_times, still_frames, image_ids, trial_keys = trials.fetch['flip_times', 'still_frame', 'still_id', dj.key]

        assert np.all([ft.size == 2 for ft in flip_times]), "Fliptimes do not have length 2"

        stimulus_onset = np.vstack(flip_times)  # columns correspond to  clear flip, onset flip
        print('Average inferred stimulus length', np.mean(stimulus_onset[1:, 0] - stimulus_onset[:-1, 1]), 's',
              flush=True)
        stimulus_onset = stimulus_onset[:, 1]

        # subtract irrelevant start time
        stimulus_onset = stimulus_onset - frame_times[0]
        frame_times = frame_times - frame_times[0]
        return frame_times, stimulus_onset, trial_keys

    @staticmethod
    def _get_behavior_timiming(key):
        fr_to_behav, nslices = (BehaviorSync() * Prepare.Galvo() & key).fetch1['frame_times', 'nslices']
        sli = IntegratedResponse._slice(key)
        return fr_to_behav.squeeze()[sli - 1::nslices]

    def _make_tuples(self, key):
        print('Populating', key)

        trace = self._get_spike_trace(key)
        frame_times, stimulus_onset, trial_keys = self._get_stimulus_timiming(key)
        frame_times = frame_times[:len(trace)]

        fr_to_behav = self._get_behavior_timiming(key)
        fr_to_behav = fr_to_behav[:len(trace)]

        intr_windows = (IntegrationWindow() & key).get_bin_borders()

        responses = integration_factory(frame_times, trace)

        # minmal and maximal time of onset for trial to lie full in recoreded frames
        t_min, t_max = frame_times[0] - intr_windows[0], frame_times[-1] - intr_windows[-1]
        droptrials = (stimulus_onset > t_max) | (stimulus_onset < t_min)
        if np.any(droptrials):
            print('Dropping', droptrials.sum(), 'trials because they are outside the recorded window')
            stimulus_onset = stimulus_onset[~droptrials]
            trial_keys = [tk for tk, drop in zip(trial_keys, droptrials) if not drop]

        assert len(stimulus_onset) == len(trial_keys), 'len of onset and trial_keys does not match'

        n = len(stimulus_onset)

        self.insert1(key)
        trial_table = self.Trial()
        for i, on, k in zip(count(), stimulus_onset, trial_keys):
            print('\r{}/{}'.format(i, n), end='', flush=True)
            trial_table.insert1(
                dict(key,
                     response=np.atleast_2d(responses(on + intr_windows)),
                     behavior_bins=np.atleast_2d(np.interp(on + intr_windows, frame_times, fr_to_behav)),
                     **k)
            )


# @schema
# class StillImageRespo(dj.Computed):
#     definition = """
#
#     """



schema.spawn_missing_classes()
