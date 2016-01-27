import datajoint as dj
from djaddon import gitlog
import numpy as np
import c2s
import matplotlib.pyplot as plt
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
        for i, (i1, i2, j1, j2, mask) in enumerate(zip(*selection.fetch['rstart','rend','cstart','cend','mask'])):
            masks[i1-1:i2,j1-1:j2,i]  = mask
        return masks

    def plot_masks(self, key):
        masks = self.load_masks(key) 
        amask = (masks>0).sum(axis=2)
        v = np.unique(amask)
        masked = np.ma.masked_where(masks == 0, masks)
        
        for i in range(masks.shape[2]):
            fig, ax = plt.subplots()
            ax.contour(amask, v)
            ax.imshow(masked[::-1,:,i], cmap=plt.cm.BuPu)
            ax.set_aspect(1)
            ax.axis('tight')
            plt.show()
            exit()
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
        data['calcium'] = (MaskAverageTrace()& key).fetch1['trace'].squeeze()
        data = c2s.preprocess([data], fps=data['fps'])
        data[0]['calcium'] = np.atleast_2d(data[0]['calcium'])
        data = c2s.predict(data)
        data = data[0]
        data['rate'] = data.pop('predictions').squeeze()
        data.pop('calcium')
        data.pop('fps')
        self.insert1(data)

