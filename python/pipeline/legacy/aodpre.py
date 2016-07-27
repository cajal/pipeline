
from warnings import warn
import numpy as np
from . import pre

import datajoint as dj
schema = dj.schema('pipeline_aod_preprocessing', locals())



@schema
class Spikes(dj.Computed):
    definition = ...

    def _make_tuples(self, key):
        raise NotImplementedError("""This is an old style part table inherited from matlab.
        call worker on dj.ExtracSpikes. This will call make_tuples on this class. Do not
        call make_tuples in pre.Spikes!
        """)

    def make_tuples(self, key):

        dt = 1/(Set() & key).fetch1['sampling_rate']
        X = (Trace() & key).project('trace').fetch.as_dict()
        for x in X:
            x = (pre.SpikeInference() & key).infer_spikes([x], dt, trace_name='trace')
            self.insert1(dict(key, **x[0]))


@schema
class ExtractSpikes(dj.Computed):
    definition = ...

    @property
    def populated_from(self):
        # Segment and SpikeInference will be in the workspace if they are in the database
        return  ComputeTraces() * pre.SpikeInference() & 'language="python"'

    def _make_tuples(self, key):
        self.insert1(key)
        Spikes().make_tuples(key)