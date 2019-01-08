""" Test suite for pre processing routines."""
import numpy as np
from numpy.testing import assert_allclose
from pipeline.utils import galvo_corrections

##### Motion correction

def test_motion_correction_type():
    # Double to double
    test_scan = np.double(np.arange(16).reshape([4, 4]))
    result = galvo_corrections.correct_motion(test_scan, xy_motion=np.array([0.1, -0.1]))
    assert (result.dtype == np.double), 'Motion correction is changing the scan dtype'

    #int to double
    test_scan = np.arange(16).reshape([4, 4])
    result = galvo_corrections.correct_motion(test_scan, xy_motion=np.array([0.1, -0.1]))
    assert (result.dtype == np.double), 'Motion correction is not changing the scan ' \
                                        'dtype from int64 to double'

def test_motion_correction_is_accurate():
    test_scan = np.double(np.arange(36).reshape([6, 6]))
    result = galvo_corrections.correct_motion(test_scan, xy_motion=np.array([0.1, -0.1]))
    desired_result = [[6.5, 7.5, 8.5, 9.5], [12.5, 13.5, 14.5, 15.5],
                      [18.5, 19.5, 20.5, 21.5], [24.5, 25.5, 26.5, 27.5]]

    assert_allclose(result[1:-1,1:-1], desired_result, err_msg='Motion correction is not '
                                                               'accurate enough')

def test_motion_correction_with_ndimensional_input():
    test_scan = np.double(np.arange(128).reshape([4, 4, 2, 2, 2]))
    xy_motion = np.reshape(np.arange(16)/10, [2,1,2,2,2])
    result = galvo_corrections.correct_motion(test_scan, xy_motion=xy_motion)
    desired_first_image = [[25.6, 33.6, 41.6, 49.6], [57.6, 65.6, 73.6, 81.6],
                           [89.6, 97.6, 105.6, 113.6], [96, 104, 112, 120]]

    assert_allclose(result[:, :, 0, 0, 0], desired_first_image,
                    err_msg='Motion correction is not properly handling n-dimensional '
                            'scans')

def test_motion_correction_no_motion():
    test_scan = np.double(np.arange(1, 17).reshape([4, 4]))
    result = galvo_corrections.correct_motion(test_scan, xy_motion=np.array([0, 0]))
    assert_allclose(result, np.arange(1, 17).reshape([4, 4]),
                    err_msg='Motion correction with zero xy_motion changes the result')

def test_motion_correction_list_input():
    test_scan = np.double(np.arange(36).reshape([6, 6]))
    result = galvo_corrections.correct_motion(test_scan, [0.1, -0.1])
    desired_result = [[6.5, 7.5, 8.5, 9.5], [12.5, 13.5, 14.5, 15.5],
                      [18.5, 19.5, 20.5, 21.5], [24.5, 25.5, 26.5, 27.5]]

    assert_allclose(result[1:-1, 1:-1], desired_result,
                    err_msg='Motion correction can not handle xy_motion as a list.')

def test_motion_correction_nan_xy_motion():
    test_scan = np.double(np.arange(16).reshape([4, 4]))
    result = galvo_corrections.correct_motion(test_scan,
                                              xy_motion=np.array([np.nan, 0.1]))
    assert_allclose(result, np.arange(16).reshape([4, 4]),
                    err_msg='Motion correction cannot handle nan in xy_motion')

def test_motion_correction_not_in_place():
    test_scan = np.double(np.arange(16).reshape([4, 4]))
    result = galvo_corrections.correct_motion(test_scan, xy_motion=[0.1, -0.1],
                                              in_place=False)
    assert_allclose(test_scan, np.arange(16).reshape([4, 4]),
                    err_msg='Motion correction is not creating a copy of the scan when '
                            'asked to (in_place=False)')


##### Raster correction

def test_raster_correction_is_accurate():
    test_scan = np.double(np.arange(128).reshape([4,4,2,2,2]))
    result = galvo_corrections.correct_raster(test_scan, raster_phase=0.5, fill_fraction=1)
    # taken from Matlab
    desired_first_image = [[6.542, 15.917, 22.938, 60],[60, 33.06, 40.083, 49.457],
                           [70.543, 79.917, 86.938, 60],[60, 97.062, 104.083, 113.457]]

    assert_allclose(result[:,:,0,0,0], desired_first_image, rtol=0.01,
                    err_msg='Raster correction is not accurate enough')

def test_raster_correction_type():
    # Double to double
    test_scan = np.double(np.arange(128).reshape([4, 4, 2, 2, 2]))
    result = galvo_corrections.correct_raster(test_scan, raster_phase=0.5, fill_fraction=1)
    assert (result.dtype == np.double), 'Raster correction is changing the scan dtype'

    #int to double
    test_scan = np.arange(128).reshape([4, 4, 2, 2, 2])
    result = galvo_corrections.correct_raster(test_scan, raster_phase=0.5, fill_fraction=1)
    assert (result.dtype == np.double), 'Raster correction is not changing the scan ' \
                                        'dtype from int64 to double'

def test_raster_correction_with_smaller_input():
    test_scan = np.double(np.arange(32).reshape([4, 4, 2]))
    result = galvo_corrections.correct_raster(test_scan, raster_phase=0.5, fill_fraction=1)
    assert (result.ndim == 3), 'Dimensions of result do not match those of the scan in ' \
                               'raster correction'

def test_raster_correction_with_zero_raster_phase():
    test_scan = np.double(np.arange(128).reshape([4, 4, 2, 2, 2]))
    result = galvo_corrections.correct_raster(test_scan, raster_phase=0, fill_fraction=1)
    desired_first_image = [[0, 8, 16, 24], [32, 40, 48, 56], [64, 72, 80, 88],
                           [96, 104, 112, 120]]

    assert_allclose(desired_first_image, result[:, :, 0, 0, 0],
                    err_msg='Raster correction with zero raster_phase changes the result')

def test_raster_correction_nonsquare_images():
    test_scan = np.double(np.arange(24).reshape([3, 4, 2]))
    result = galvo_corrections.correct_raster(test_scan, raster_phase=0.5, fill_fraction=1)
    assert (result.shape == (3,4,2)), 'Shape of result is different from shape of scan ' \
                                      'in raster correction'

def test_raster_correction_not_in_place():
    test_scan = np.double(np.arange(128).reshape([4, 4, 2, 2, 2]))
    result = galvo_corrections.correct_raster(test_scan, raster_phase=0.5, fill_fraction=1,
                                              in_place=False)
    assert_allclose(test_scan, np.arange(128).reshape([4, 4, 2, 2, 2]),
                    err_msg='Raster correction is not creating a copy of the scan when '
                            'asked to (in_place=False)')

if __name__ == '__main__':
    import nose
    nose.main()
