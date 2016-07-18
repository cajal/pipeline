import sh
from scipy import ndimage
from warnings import warn
from sklearn.metrics import roc_curve
import datajoint as dj
from . import rf, trippy
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from pprint import pprint
import pandas as pd
import os

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
        call populate on dj.ExtracSpikes. This will call make_tuples on this class. Do not
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


    def mask_area_hists(self, outdir='./'):
        # TODO: plot against firing rate once traces are repopulated
        with sns.axes_style('ticks'):
            fig, ax = plt.subplots()

        for key in (self.project() * SegmentMethod() & dict(method_name='nmf')).fetch.as_dict:
            area_per_pixel = np.prod((ScanInfo() & key).fetch1['um_width','um_height']) / \
                                np.prod((ScanInfo() & key).fetch1['px_width','px_height'])
            areas = np.array([pxs*area_per_pixel for pxs in map(len, (SegmentMask() & key).fetch['mask_pixels'])])
            ax.hist(areas, bins=20, alpha=.5, lw=0, label="A{animal_id}S{session}:{scan_idx}".format(**key))
        ax.legend()
        sns.despine(fig)
        plt.show()


    def plot_NMF_ROIs(self, outdir='./'):
        sns.set_context('paper')
        theCM = sns.blend_palette(['lime', 'gold', 'deeppink'], n_colors=10)  # plt.cm.RdBu_r

        for key in (self.project() * SegmentMethod()*SpikeInference() & dict(short_name='stm', method_name='nmf')).fetch.as_dict:
            mask_px, mask_w, ca, sp = (SegmentMask()*Trace()*Spikes() & key).fetch.order_by('mask_id')['mask_pixels', 'mask_weights', 'ca_trace', 'spike_trace']

            template = np.stack([normalize(bugfix_reshape(t)[..., key['slice']-1].squeeze())
                                 for t in (ScanCheck() & key).fetch['template']], axis=2).mean(axis=2) # TODO: remove bugfix_reshape once djbug #191 is fixed

            d1, d2 = tuple(map(int, (ScanInfo() & key).fetch1['px_height', 'px_width']))
            masks = Segment.reshape_masks(mask_px, mask_w, d1, d2)
            gs = plt.GridSpec(6,1)
            try:
                sh.mkdir('-p', os.path.expanduser(outdir) + '/scan_idx{scan_idx}/slice{slice}'.format(**key))
            except:
                pass
            for cell, (ca_trace, sp_trace) in enumerate(zip(ca, sp)):
                with sns.axes_style('white'):
                    fig = plt.figure(figsize=(6,8))
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
                fig.suptitle("animal_id {animal_id}:session {session}:scan_idx {scan_idx}:{method_name}:slice{slice}:cell{cell}".format(cell=cell+1, **key))
                fig.tight_layout()

                plt.savefig(outdir + "/scan_idx{scan_idx}/slice{slice}/cell{cell:03d}_animal_id_{animal_id}_session_{session}.png".format(cell=cell+1, **key))
                plt.close(fig)

schema.spawn_missing_classes()