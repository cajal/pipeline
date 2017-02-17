""" Test suite for pre processing routines."""
#from nose.tools import assert_raises, assert_equal, \
#    assert_false, assert_true, assert_list_equal, \
#    assert_tuple_equal, assert_dict_equal, raises
import numpy as np
from pipeline.utils import galvo_corrections

def test_motioncorrect():
    """test motion correction"""

    img = np.random.randn(50, 50)
    xymotion = [2, 5]
    newim = galvo_corrections.correct_motion(img, xymotion)

    d = np.abs(np.amax(img[xymotion[1]:, xymotion[0]:] - newim[:-xymotion[1], :-xymotion[0]]))

    assert(0 < d < 10e-15)

def test_raster_correction_is_accurate():
    test_scan = np.double(np.arange(128)).reshape([4,4,2,2,2])
    result = galvo_corrections.correct_raster(test_scan, raster_phase=0.5, fill_fraction=1)

    # taken from Matlab
    desired_first_image = [[6.542, 15.917, 22.938, 60],[60, 33.06, 40.083, 49.457],
                            [70.543, 79.917, 86.938, 60],[60, 97.062, 104.083, 113.457]]

    np.testing.assert_almost_equal(desired_first_image, result[:,:,0,0,0], decimal = 2)

def test_raster_correction_type():
    # Double to double
    test_scan = np.double(np.arange(128)).reshape([4, 4, 2, 2, 2])
    result = galvo_corrections.correct_raster(test_scan, raster_phase=0.5, fill_fraction=1)
    assert(result.dtype == np.double)

    #int to double
    test_scan = np.arange(128).reshape([4, 4, 2, 2, 2])
    result = galvo_corrections.correct_raster(test_scan, raster_phase=0.5, fill_fraction=1)
    assert(result.dtype == np.double)

def test_raster_correction_with_smaller_input():
    test_scan = np.double(np.arange(32).reshape([4, 4, 2]))
    result = galvo_corrections.correct_raster(test_scan, raster_phase=0.5, fill_fraction=1)
    assert (result.ndim == 3)

def test_raster_correction_with_zero_raster_phase():
    test_scan = np.double(np.arange(128)).reshape([4, 4, 2, 2, 2])
    result = galvo_corrections.correct_raster(test_scan, raster_phase=0, fill_fraction=1)
    desired_first_image = [[0, 8, 16, 24], [32, 40, 48, 56], [64, 72, 80, 88],
                           [96, 104, 112, 120]]

    np.testing.assert_almost_equal(desired_first_image, result[:, :, 0, 0, 0])

def test_raster_correction_nonsquare_images():
    test_scan = np.double(np.arange(24).reshape([3, 4, 2]))
    result = galvo_corrections.correct_raster(test_scan, raster_phase=0.5, fill_fraction=1)
    assert (result.shape == (3,4,2))

def test_raster_correction_in_place():
    test_scan = np.double(np.arange(128)).reshape([4, 4, 2, 2, 2])
    result = galvo_corrections.correct_raster(test_scan, raster_phase=0.5, fill_fraction=1,
                                              in_place=False)

    np.testing.assert_equal(test_scan, np.double(np.arange(128)).reshape([4, 4, 2, 2, 2]))

if __name__ == '__main__':
    import nose
    nose.main()
