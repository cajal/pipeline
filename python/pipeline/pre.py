import datajoint as dj
from djaddon import gitlog, hdf5
import numpy as np
import c2s
import matplotlib.pyplot as plt
import seaborn as sns

schema = dj.schema('pipeline_preprocessing', locals())


@schema
class AlignMotion(dj.Imported):
    definition = ...

    def _make_tuples(self, key):
        raise NotImplementedError("This schema is populated from Matlab")


@schema
class AlignRaster(dj.Computed):
    definition = ...

    def _make_tuples(self, key):
        raise NotImplementedError("This schema is populated from Matlab")


@schema
class Check(dj.Imported):
    definition = ...

    def _make_tuples(self, key):
        raise NotImplementedError("This schema is populated from Matlab")


@schema
class ExtractSpike(dj.Computed):
    definition = ...

    def _make_tuples(self, key):
        raise NotImplementedError("This schema is populated from Matlab")


@schema
class ManualSegment(dj.Imported):
    definition = ...

    def _make_tuples(self, key):
        raise NotImplementedError("This schema is populated from Matlab")


@schema
class ScanCheck(dj.Imported):
    definition = ...

    def _make_tuples(self, key):
        raise NotImplementedError("This schema is populated from Matlab")


@schema
class ScanInfo(dj.Imported):
    definition = ...

    def _make_tuples(self, key):
        raise NotImplementedError("This schema is populated from Matlab")


@schema
class Settings(dj.Lookup):
    definition = ...


@schema
class NMFSegment(dj.Imported):
    definition = ...

    def _make_tuples(self, key):
        raise NotImplementedError('This table is populated from matlab.')


@schema
class SegmentationTile(dj.Computed):
    definition = ...

    def _make_tuples(self, key):
        raise NotImplementedError('This table is populated from matlab.')


@schema
class SelectedMask(dj.Computed):
    definition = ...

    def _make_tuples(self, key):
        raise NotImplementedError('This table is populated from matlab.')

    def load_masks(self, key):
        d1, d2 = (ScanInfo() & key).fetch1['px_height', 'px_width']
        selection = (SegmentationTile() & self & key).project('mask')
        masks = np.zeros((d1, d2, len(selection)))
        for i, (i1, i2, j1, j2, mask) in enumerate(zip(*selection.fetch['rstart', 'rend', 'cstart', 'cend', 'mask'])):
            masks[i1 - 1:i2, j1 - 1:j2, i] = mask
        return masks

    def plot_masks(self, key, savedir='./'):
        if savedir[-1] != '/': savedir += '/'
        masks = self.load_masks(key)
        theCM = plt.cm.RdBu_r
        # theCM._init()
        # alphas = np.abs(np.linspace(0, 1.0, theCM.N))
        # theCM._lut[:-3,-1] = alphas

        d1, d2, frames = masks.shape
        xaxis, yaxis = np.arange(d2), np.arange(d1)
        sns.set_style('white')
        for cell in range(frames):
            print('Plotting cell %i/%i' % (cell + 1, frames))
            fig, ax = plt.subplots()
            for i in range(frames):
                ma = masks[..., i] / masks[..., i].sum()
                m = ma.ravel()[np.argsort(-ma.ravel())]
                mc = np.cumsum(m)
                th = m[mc > 0.9].max()
                if cell == i:
                    cell_ma = ma < th
                ax.contour(xaxis, yaxis, ma >= th, [.5], colors='silver')

            m = np.ma.masked_where((masks[..., cell] == 0) | cell_ma, masks[..., cell])
            ax.imshow(m, cmap=theCM, zorder=10)
            ax.set_aspect(1)
            ax.axis('tight')
            ax.axis('off')
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            fig.savefig(savedir + 'cell%03i.png' % cell)
            plt.close(fig)


@schema
class MaskAverageTrace(dj.Computed):
    definition = ...

    def _make_tuples(self, key):
        raise NotImplementedError('This table is populated from matlab.')


@schema
class Tesselation(dj.Manual):
    definition = ...


@schema
@gitlog
class STMSpikeRate(dj.Computed):
    definition = """
    # inferred spike rate trace

    ->MaskAverageTrace
    ---
    rate          : longblob  # inferred spikes
    """

    def _make_tuples(self, key):
        data = (MaskAverageTrace() * ScanInfo() & key).project('fps').fetch1()
        data['calcium'] = (MaskAverageTrace() & key).fetch1['trace'].squeeze()
        data = c2s.preprocess([data], fps=data['fps'])
        data[0]['calcium'] = np.atleast_2d(data[0]['calcium'])
        data = c2s.predict(data)
        data = data[0]
        data['rate'] = data.pop('predictions').squeeze()
        data.pop('calcium')
        data.pop('fps')
        self.insert1(data)

@schema
class Segmentation(dj.Imported):
    definition = ...

@schema
class SpikeInference(dj.Lookup):
    definition = ...

@schema
class Spikes(dj.Computed):
    definition = ...

@schema
class Template(dj.Imported):
    definition = ...

@schema
class Trace(dj.Imported):
    definition = ...

