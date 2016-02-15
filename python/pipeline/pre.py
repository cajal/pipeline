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

def normalize(img):
    return (img - img.min())/(img.max()-img.min())

def bugfix_reshape(a):
    return a.ravel(order='C').reshape(a.shape, order='F')

@schema
class SpikeInference(dj.Lookup):
    definition = ...

    def infer_spikes(self, X, dt, trace_name='ca_trace'):
        assert self.fetch1['language'] == 'python', "This tuple cannot be computed in python."
        fps = 1 / dt
        spike_rates = []
        N = len(X)
        for i, trace in enumerate(X):
            print('Predicting trace %i/%i' % (i+1,N))
            trace['calcium'] = trace.pop(trace_name).T
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
        call worker on dj.ExtracSpikes. This will call make_tuples on this class. Do not
        call make_tuples in pre.Spikes!
        """)

    def make_tuples(self, key):
        dt = 1/(ScanInfo() & key).fetch1['fps']
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
        return ExtractTraces() * SpikeInference() & dict(language='python')

    def _make_tuples(self, key):
        self.insert1(key)
        Spikes().make_tuples(key)


@schema
class Segment(dj.Imported):
    definition = ...

    def _make_tuples(self, key):
        raise NotImplementedError('This table is populated from matlab')

    @staticmethod
    def reshape_masks(mask_pixels, mask_weights, px_height, px_width):
        ret = np.zeros((px_height, px_width, len(mask_pixels)))
        for i, (mp, mw)  in enumerate(zip(mask_pixels, mask_weights)):
            mask = np.zeros(px_height * px_width)
            mask[mp.squeeze().astype(int) - 1] = mw.squeeze()
            ret[..., i] = mask.reshape(px_height, px_width, order='F')
        return ret

    # def load_masks_with_traces(self, key):
    #     d1, d2 = tuple(map(int, (ScanInfo() & key).fetch1['px_height', 'px_width']))
    #
    #     masks = np.zeros((d2, d1, len(SegmentMask() & key)))
    #     traces = []
    #
    #     for i, mask_dict in enumerate((SegmentMask()*Trace() & key).fetch.as_dict()):
    #         mask = np.zeros(d1 * d2, )
    #         mask[mask_dict['mask_pixels'].squeeze().astype(int) - 1] = mask_dict['mask_weights'].squeeze()
    #         masks[..., i] = mask.reshape(d2, d1, order='F')
    #         traces.append(mask_dict['ca_trace'].squeeze())
    #     return masks, traces
    #
    # def plot_masks_vs_manual(self):
    #
    #     sns.set_context('notebook')
    #     y = np.arange(.2, 1, .2)
    #     theCM = sns.blend_palette(['silver', 'steelblue', 'orange'], n_colors=len(y))  # plt.cm.RdBu_r
    #
    #     for key in (self.project() * SegmentMethod() - dict(
    #             method_name='manual') & ManualSegment().project()).fetch.as_dict:
    #         with sns.axes_style('white'):
    #             fig, ax = plt.subplots(figsize=(8, 8))
    #
    #
    #         ground_truth = bugfix_reshape((ManualSegment() & key).fetch1['mask'])   # TODO: remove bugfix_reshape once djbug #191 is fixed
    #
    #         template = np.stack([normalize(bugfix_reshape(t)[..., key['slice']-1].squeeze())
    #                              for t in (ScanCheck() & key).fetch['template']], axis=2).mean(axis=2) # TODO: remove bugfix_reshape once djbug #191 is fixed
    #
    #         masks,_ = self.load_masks_with_traces(key)
    #         frames = masks.shape[2]
    #
    #         for cell in range(frames):
    #             ma = masks[..., cell].ravel()
    #             ma.sort()
    #             cdf = ma.cumsum()
    #             cdf = cdf / cdf[-1]
    #             th = np.interp(y, cdf, ma)
    #             ax.contour(masks[..., cell], th, colors=theCM)
    #
    #         ax.imshow(template, cmap=plt.cm.gray)
    #         ax.contour(ground_truth, [.5], colors='deeppink')
    #         ax.set_title("animal_id {animal_id}:session {session}:scan_idx {scan_idx}:{method_name}:slice{slice}".format(**key))
    #         fig.tight_layout()

    def plot_single_ROIs(self, outdir='./'):
        sns.set_context('paper')
        theCM = sns.blend_palette(['lime', 'gold', 'deeppink'], n_colors=10)  # plt.cm.RdBu_r

        for key in ((self.project() * SegmentMethod()*SpikeInference() & dict(short_name='stm'))
                        - dict(method_name='manual') & ManualSegment().project()).fetch.as_dict:
            mask_px, mask_w, ca, sp = (SegmentMask()*Trace()*Spikes() & key).fetch['mask_pixels', 'mask_weights', 'ca_trace', 'spike_trace']

            template = np.stack([normalize(bugfix_reshape(t)[..., key['slice']-1].squeeze())
                                 for t in (ScanCheck() & key).fetch['template']], axis=2).mean(axis=2) # TODO: remove bugfix_reshape once djbug #191 is fixed

            d1, d2 = tuple(map(int, (ScanInfo() & key).fetch1['px_height', 'px_width']))
            masks = Segment.reshape_masks(mask_px, mask_w, d1, d2)
            gs = plt.GridSpec(6,1)
            for cell, (ca_trace, sp_trace) in enumerate(zip(ca, sp)):
                with sns.axes_style('white'):
                    fig = plt.figure(figsize=(4,6))
                    ax_image = fig.add_subplot(gs[2:,:])

                with sns.axes_style('ticks'):
                    ax_ca = fig.add_subplot(gs[0,:])
                    ax_sp = fig.add_subplot(gs[1,:], sharex=ax_ca)
                ax_ca.plot(ca_trace,'green', lw=1)
                ax_sp.plot(sp_trace,'k',lw=1)

                ax_image.imshow(template, cmap=plt.cm.gray)
                ax_image.contour(masks[..., cell], colors=theCM, zorder=10    )
                sns.despine(ax=ax_ca)
                sns.despine(ax=ax_sp)
                ax_ca.axis('tight')
                ax_sp.axis('tight')
                fig.suptitle("animal_id {animal_id}:session {session}:scan_idx {scan_idx}:{method_name}:slice{slice}".format(**key))
                fig.tight_layout()
                plt.savefig(outdir + "/" + "cell{cell}_animal_id_{animal_id}_session_{session}_scan_idx_{scan_idx}_{method_name}_slice_{slice}.pdf".format(cell=cell, **key))
                plt.close(fig)


    # def plot_ROC_curves(self):
    #     """
    #     Takes all masks from an NMF segmentation, L1 normalizes them, computes a MAX image from it and uses that to
    #     plot and ROC curve using the manual segmentations as ground truth.
    #     """
    #
    #     sns.set_context('notebook')
    #
    #     with sns.axes_style('whitegrid'):
    #         fig, ax = plt.subplots(figsize=(8, 8))
    #     for key in (self.project() * SegmentMethod() - dict(
    #             method_name='manual') & ManualSegment().project()).fetch.as_dict:
    #         ground_truth = bugfix_reshape((ManualSegment() & key).fetch1['mask'])   # TODO: remove bugfix_reshape once djbug #191 is fixed
    #         masks, _ = self.load_masks_with_traces(key)
    #         masks /= masks.sum(axis=0).sum(axis=0)[None, None, :]
    #         masks = masks.max(axis=2)
    #         fpr, tpr, _ = roc_curve(ground_truth.ravel(), masks.ravel())
    #         ax.plot(fpr, tpr,
    #                 label="animal_id {animal_id}:session {session}:scan_idx {scan_idx}:{method_name}:slice{slice}".format(**key))
    #         ax.set_xlabel('false positives rate')
    #         ax.set_ylabel('true positives rate')
    #         ax.legend(loc='lower right')
