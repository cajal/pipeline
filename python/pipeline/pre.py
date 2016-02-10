from scipy import ndimage
from warnings import warn

from sklearn.metrics import roc_curve

import datajoint as dj
from . import rf, trippy
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from pprint import pprint

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
            data = c2s.predict(data, verbosity=0)
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
        return Segment() * SpikeInference() & rf.Sync() & dict(language='python')

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

        masks = np.zeros((d2, d1, len(SegmentMask() & key)))
        for i, mask_dict in enumerate((SegmentMask() & key).fetch.as_dict()):
            mask = np.zeros(d1 * d2,)
            mask[mask_dict['mask_pixels'].squeeze().astype(int) - 1] = mask_dict['mask_weights'].squeeze()
            masks[..., i] = mask.reshape(d2, d1, order='F')
        return masks

    def plot_masks_vs_manual(self):

        sns.set_context('notebook')
        y = np.arange(.2, 1, .2)
        theCM = sns.blend_palette(['silver', 'steelblue', 'orange'], n_colors=len(y))  # plt.cm.RdBu_r


        for key in (self.project()*SegmentMethod() - dict(method_name='manual') & ManualSegment().project()).fetch.as_dict:
            with sns.axes_style('white'):
                fig, ax = plt.subplots(figsize=(8,8))
            ground_truth = (ManualSegment() & key).fetch1['mask'].T # TODO: remove .T once djbug #191 is fixed
            template = (ScanCheck() & key).fetch1['template'].T # TODO: remove .T once djbug #191 is fixed

            masks = self.load_masks(key)
            frames = masks.shape[2]

            for cell in range(frames):
                ma = masks[..., cell].ravel()
                ma.sort()
                cdf = ma.cumsum()
                cdf = cdf / cdf[-1]
                th = np.interp(y, cdf, ma)
                ax.contour(masks[..., cell], th, colors=theCM)

            ax.imshow(template, cmap=plt.cm.gray)
            ax.contour(ground_truth, [.5], colors='deeppink')
            ax.set_title("animal_id {animal_id}:session {session}:scan_idx {scan_idx}:{method_name}".format(**key))
            fig.tight_layout()

    def plot_ROC_curves(self):
        """
        Takes all masks from an NMF segmentation, L1 normalizes them, computes a MAX image from it and uses that to
        plot and ROC curve using the manual segmentations as ground truth.
        """

        sns.set_context('notebook')

        with sns.axes_style('whitegrid'):
            fig, ax = plt.subplots(figsize=(8,8))
        for key in (self.project()*SegmentMethod() - dict(method_name='manual') & ManualSegment().project()).fetch.as_dict:
            ground_truth = (ManualSegment() & key).fetch1['mask'].T # TODO: remove .T once djbug #191 is fixed
            masks = self.load_masks(key)
            masks /= masks.sum(axis=0).sum(axis=0)[None, None, :]
            masks = masks.max(axis=2)
            fpr, tpr, _ = roc_curve(ground_truth.ravel(), masks.ravel())
            ax.plot(fpr, tpr, label="animal_id {animal_id}:session {session}:scan_idx {scan_idx}:{method_name}".format(**key))
            ax.set_xlabel('false positives rate')
            ax.set_ylabel('true positives rate')
            ax.legend(loc='lower right')
