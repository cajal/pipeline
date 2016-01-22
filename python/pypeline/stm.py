import datajoint as dj
from . import nmf, rf
from functools import wraps
import numpy as np
import c2s
from djaddon import gitlog

schema = dj.schema('microns_stm', locals())


@schema
@gitlog
class SpikeRate(dj.Computed):
    definition = """
    # inferred spike rate trace
    ->nmf.MaskAverageTrace
    ---
    rate          : longblob  # inferred spikes
    """

    def _make_tuples(self, key):
        data = (nmf.MaskAverageTrace() * rf.ScanInfo() & key).fetch1()
        data['calcium'] = data.pop('trace').squeeze()
        data = c2s.preprocess([data], fps=data['fps'])
        data[0]['calcium'] = np.atleast_2d(data[0]['calcium'])
        data = c2s.predict(data)
        data = data[0]
        data['rate'] = data.pop('predictions').squeeze()
        self.insert1(data)

