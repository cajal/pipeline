"""
Analysis of visual tuning: receptive fields, tuning curves, pixelwise maps
It differs from tuning.py in that it uses the new visual stimulus schema `stimulus` whereas tuning was using `vis`
"""
import itertools
import numpy as np
import datajoint as dj
from scipy.interpolate import interp1d
from . import reso, experiment, stimulus


schema = dj.schema('pipeline_tune', locals())


@schema
class Drift(dj.Computed):
    """
    # all directional drift trials for the scan
    -> stimulus.Sync
    ---
    ndirections: tinyint  # number of unique directions
    """

    class Trial(dj.Part):
        """
        -> Drift
        drift_trial: smallint  # trial index
        ---
        -> stimulus.Trial
        direction: float  # (degrees) direction of drift
        onset: double  # (s) onset time in rf.Sync times
        offset: double  # (s) offset time in rf.Sync times
        """

    @property
    def key_source(self):
        return stimulus.Sync() & (stimulus.Trial() & (stimulus.Monet().proj('speed') & 'speed>0'))


    def _make_tuples(self, key):
        frame_times = (stimulus.Sync() & key).fetch1('frame_times').squeeze()
        direction_set = set()
        count = itertools.count()
        tuples = list()

        for trial_key in (stimulus.Trial() & stimulus.Monet().proj() & key).fetch.keys():
            onsets, directions, ori_duration, flips = (stimulus.Trial() * stimulus.Monet() & trial_key).fetch1(
                'onsets', 'directions', 'ori_on_secs', 'flip_times')
            flips = flips.squeeze()
            onsets = onsets.squeeze() + flips[0]
            directions = directions.squeeze()/np.pi*180
            direction_set.update(directions)
            tuples.extend(dict(key,
                               drift_trial=drift_trial,
                               onset=onset,
                               offset=onset+ori_duration,
                               direction=direction,
                               **trial_key)
                          for drift_trial, onset, direction in zip(count, onsets, directions))
        self.insert1(dict(key, ndirections=len(direction_set)))
        self.Trial().insert(tuples)


@schema
class Response(dj.Computed):
    """  # response to directional stimulus
    -> reso.Activity
    -> Drift
    ---
    latency : float   # latency used (ms)
    """

    class Trial(dj.Part):
        definition = """   # the response for each trial and each trace
        -> reso.Activity.Trace
        -> Drift.Trial
        ---
        response : float   #  integrated response
        """

    def _make_tuples(self, key):
        print('Directional response for ', key)
        traces, slice, trace_keys = (reso.Activity.Trace()*reso.UnitSet.Unit() & key).fetch('trace', 'slice', dj.key)
        traces = np.float64(np.stack(t.flatten() for t in traces))

        #  fetch and clean up the trace time
        trace_time = (stimulus.Sync() & key).fetch1('frame_times').squeeze()  # calcium scan frame times
        n_slices = (reso.ScanInfo() & key).fetch1('nslices')
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
        table = Response.Trial()
        for onset, offset, trial_key in zip(*(Drift.Trial() & key).fetch('onset', 'offset', dj.key)):
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


schema.spawn_missing_classes()