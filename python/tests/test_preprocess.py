from nose.tools import assert_raises, assert_equal, \
    assert_false, assert_true, assert_list_equal, \
    assert_tuple_equal, assert_dict_equal, raises
import numpy as np
from scipy.interpolate import interp1d
from pipeline.utils import galvo_corrections
from pipeline import preprocess
from tiffreader import TIFFReader


def test_motioncorrect():
    """test motion correction"""

    img = np.random.randn(50, 50)
    xymotion = [2, 5]
    newim = galvo_corrections.correct_motion(img, xymotion)

    d = np.abs(np.amax(img[xymotion[1]:, xymotion[0]:] - newim[:-xymotion[1], :-xymotion[0]]))

    assert_true(0 < d < 10e-15)


def test_rastercorrect():
    """test raster correction"""

    img = np.random.randn(128, 128, 1, 1, 10)
    raster_phase = np.pi/2
    fill_fraction = 1
    newimg = galvo_corrections.correct_raster(img, raster_phase, fill_fraction)
    reimg = galvo_corrections.correct_raster(newimg, -raster_phase, fill_fraction)

    d1 = np.mean(np.abs(newimg - img))
    d2 = np.mean(np.abs(img - reimg))

    assert_true(d1 != 0 and d2 < np.mean(np.abs(img)))
