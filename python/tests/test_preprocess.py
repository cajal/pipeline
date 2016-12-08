from nose.tools import assert_raises, assert_equal, \
    assert_false, assert_true, assert_list_equal, \
    assert_tuple_equal, assert_dict_equal, raises
import numpy as np
from pipeline.utils import galvo_corrections
from scipy.interpolate import interp1d

def test_motioncorrect():
    """test motion correction"""

    img = np.random.randn(50, 50)
    xymotion = [2, 5]
    newim = galvo_corrections.correct_motion(img, xymotion)

    d = np.abs(np.amax(img[xymotion[1]:, xymotion[0]:] - newim[:-xymotion[1], :-xymotion[0]]))

    assert_true(0 < d < 10e-15)


def test_rastercorrect():
    """test raster correction"""

    # img = np.random.randn(512, 512, 1, 1, 100)
    # raster_phase = 0.1
    # fill_fraction = 1
    from pipeline import preprocess
    from tiffreader import TIFFReader
    reader = TIFFReader('/Users/titan/data/cache/11676_4_00006_00009.tif')
    img = reader[:, :, 0, 0, 10:15]
    raster_phase = (preprocess.Prepare.Galvo() & dict(animal_id=11676, session=4, scan_idx=6)).fetch1['raster_phase']
    fill_fraction = reader.fill_fraction
    newimg = galvo_corrections.correct_raster(img, raster_phase, fill_fraction)
    reimg = galvo_corrections.correct_raster(newimg, -raster_phase, fill_fraction)

    d1 = np.mean(np.abs(newimg - img))
    d2 = np.mean(np.abs(img - reimg))

    assert_true(d1 != 0 and d2 < np.mean(np.abs(img)))
