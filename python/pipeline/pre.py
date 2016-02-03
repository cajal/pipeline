from warnings import warn

import datajoint as dj
from . import rf, trippy
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

try:
    import c2s
except:
    warn("c2s was not found. You won't be able to populate ExtracSpikes")

schema = dj.schema('pipeline_preprocessing', locals())


@schema
class SpikeInference(dj.Lookup):
    definition = ...

    def infer_spikes(self, X, dt):
        assert self.fetch1['language'] == 'python', "This tuple cannot be computed in python."
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


@schema
class Segment(dj.Imported):
    definition = ...

    def _make_tuples(self, key):
        raise NotImplementedError('This table is populated from matlab')

    def load_masks(self, key):
        d1, d2 = tuple(map(int, (ScanInfo() & key).fetch1['px_height', 'px_width']))

        masks = np.zeros((d1, d2, len(SegmentMask() & key)))
        for i, mask_dict in enumerate((SegmentMask() & key).fetch.as_dict()):
            mask = np.zeros(d1 * d2)
            mask[mask_dict['mask_pixels'].squeeze().astype(int) - 1] = mask_dict['mask_weights']

            masks[..., i] = mask.reshape(d1, d2, order='F')
        return masks

    def plot_masks(self, key, savedir='./'):
        """
        Plot the segmentation masks

        :param key:
        :param savedir:
        """
        assert (self * SegmentMethod() & key).fetch1['method_name'] == 'nmf', \
                            "Only work for nmf segmentation at the moment"

        if savedir[-1] != '/': savedir += '/'
        masks = self.load_masks(key)

        d1, d2, frames = masks.shape
        xaxis, yaxis = np.arange(d2), np.arange(d1)
        sns.set_style('white')
        fig, ax = plt.subplots(figsize=(7, 7), dpi=400)
        y = np.arange(.2, 1, .2)
        theCM = sns.blend_palette(['silver', 'steelblue', 'orange'], n_colors=len(y))  # plt.cm.RdBu_r
        for cell in range(frames):
            ma = masks[..., cell].ravel()
            ma.sort()
            cdf = ma.cumsum()
            cdf = cdf / cdf[-1]
            th = np.interp(y, cdf, ma)
            ax.contour(xaxis, yaxis, masks[..., cell], th, colors=theCM)

        ax.set_title(' '.join(['%s: %s' % (str(k), str(v)) for k, v in key.items()]), fontsize=8, fontweight='bold')
        ax.set_aspect(1)
        ax.axis('tight')
        ax.axis('off')
        fig.savefig(savedir + '__'.join(['%s_%s' % (str(k), str(v)) for k, v in key.items()]) + '.png')
