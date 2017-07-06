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
        z = (ExtractRaw.GalvoROI() * MaskCoordinates() & key).fetch['zloc']
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
        print('Populating', key)
        d1, d2 = tuple(map(int, (Prepare.Galvo() & key).fetch1['px_height', 'px_width']))

        if Prepare.Meso() & key:
            v1 = (AreaMask().proj('mask', scan_idx2='scan_idx') & key & dict(area='V1')).fetch1['mask']
            lm = (AreaMask().proj('mask', scan_idx2='scan_idx') & key & dict(area='LM')).fetch1['mask']
        else:
            v1, lm = (AreaBorder().proj('V1_mask', 'LM_mask', dummy='scan_idx') & key).fetch1['V1_mask', 'LM_mask']

        mask_key, weight, px, sli = (ExtractRaw.GalvoROI() & key).fetch1[dj.key, 'mask_weights', 'mask_pixels', 'slice']

        mask = ExtractRaw.GalvoROI.reshape_masks([px], [weight], d1, d2).squeeze()
        m = mask/mask.sum()

        I, J = map(np.transpose, np.meshgrid(*[np.arange(i) for i in lm.shape]))
        loc = int((I * m).sum()), int((J * m).sum())

        del key['axis']
        del key['dummy']

        if lm[loc]:
            self.insert1(dict(key, brain_area='LM', **mask_key))
        elif v1[loc]:
            self.insert1(dict(key, brain_area='V1', **mask_key))
        else:
            raise ValueError('mask seems to be neither in V1 or LM')

