import datajoint as dj
from .experiment import BrainArea, Layer
from .preprocess import ExtractRaw, Prepare, Slice, MaskCoordinates
import numpy as np

schema = dj.schema('pipeline_map', locals())
schema.spawn_missing_classes()


@schema
class LayerMembership(dj.Computed):
    definition = """
    -> ExtractRaw.GalvoROI
    -> Slice
    -> MaskCoordinates
    ---
    -> Layer
    """

    key_source = ExtractRaw.GalvoROI() * Slice() & dict(extract_method=2)


    def _make_tuples(self, key):
        z = (ExtractRaw.GalvoROI() * MaskCoordinates() & key).fetch('zloc')
        self.insert1(dict(key, layer=str(Layer().get_layers(z))))


@schema
class AreaMembership(dj.Computed):
    definition = """
    # brain area membership of cells

    -> ExtractRaw.GalvoROI
    ---
    -> BrainArea
    """

    @property
    def key_source(self):
        return ExtractRaw.GalvoROI() * AreaBorder().proj(dummy='scan_idx') & dict(extract_method=2)

    def _make_tuples(self, key):
        d1, d2 = tuple(map(int, (Prepare.Galvo() & key).fetch1('px_height', 'px_width'))

        keys, weights, px, slices = (ExtractRaw.GalvoROI() & key).fetch(dj.key, 'mask_weights', 'mask_pixels', 'slice')

        v1, lm = (AreaBorder().proj('V1_mask', 'LM_mask', dummy='scan_idx') & key).fetch1('V1_mask', 'LM_mask')
        masks = ExtractRaw.GalvoROI.reshape_masks(px, weights, d1, d2)

        I, J = map(np.transpose, np.meshgrid(*[np.arange(i) for i in lm.shape]))
        locs = [(int((I * m).sum()), int((J * m).sum())) \
                for m in map(lambda x: x / x.sum(), masks.transpose([2, 0, 1]))]

        del key['axis']
        del key['dummy']

        self.insert(dict(key, brain_area='LM', **k) for k, loc in zip(keys, locs) if lm[loc])
        self.insert(dict(key, brain_area='V1', **k) for k, loc in zip(keys, locs) if v1[loc])
