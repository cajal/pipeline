import datajoint as dj
from . import rf
from djaddon import gitlog, hdf5
import numpy as np
import c2s
import matplotlib.pyplot as plt
import seaborn as sns

schema = dj.schema('pipeline_preprocessing', locals())


@schema
class SpikeInference(dj.Lookup):
    definition = ...

    def infer_spikes(self, X, dt):
        method = self.fetch1['short_name']
        assert method == 'stm', "Everything except 'stm' is computed in matlab"
        fps = 1 / dt
        spike_rates = []
        for i, trace in enumerate(X):
            trace['calcium'] = trace.pop('ca_trace').T
            trace['fps'] = fps

            data = c2s.preprocess([trace], fps=fps)
            data = c2s.predict(data)
            data[0]['spike_trace'] = data[0].pop('predictions').T
            data[0].pop('calcium')
            data[0].pop('fps')
            spike_rates.append(data[0])
        return spike_rates


@schema
class Spikes(dj.Computed):
    definition = ...

    def _make_tuples(self, key):
        raise NotImplementedError("""This is an old style part table inherited from matlab.
        call populate on dj.ExtracSpikes. This will call make_tuples on this class. Do not
        call make_tuples in pre.Spikes!
        """)

    def make_tuples(self, key):
        times = (rf.Sync() & key).fetch1['frame_times']
        dt = np.median(np.diff(times))
        X = (Trace() & key).project('ca_trace').fetch.as_dict()
        X = (SpikeInference() & key).infer_spikes(X, dt)
        for x in X:
            self.insert1(dict(key, **x))


@schema
class ExtractSpikes(dj.Computed):
    definition = ...

    @property
    def populated_from(self):
        # Segment and SpikeInference will be in the workspace if they are in the database
        return Segment() * SpikeInference() & dict(short_name='stm')

    def _make_tuples(self, key):
        self.insert1(key)
        Spikes().make_tuples(key)
