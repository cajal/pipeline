""" Utilities for motion and raster correction of resonant scans. """
from ..exceptions import PipelineException
from scipy import interpolate  as interp
from scipy import signal
import numpy as np


def test(scan, template, smooth_shifts=False, smoothing_window_size=5, max_fraction_yshift=0.10, max_fraction_xshift=0.10):
    """ Compute shifts in x and y for rigid subpixel motion correction.

    Returns the number of pixels that each image in the scan was to the right (x_shift)
    or below (y_shift) the template. Negative shifts mean the image was to the left or
    above the template.

    :param np.array template: 2-d template image. Each frame in scan is aligned to this.
    :param np.array scan: 2 or 3-dimensional scan (image_height, image_width[, num_frames]).
    :param bool smooth_shifts: If True, smooth the timeseries of shifts.
    :param int smoothing_window_size: Size of the Hann window used for smoothing.
    :param max_fraction_yshift: Maximum allowed motion in y as a fraction of pixels in y.
    :param max_fraction_xshift: Maximum allowed motion in x as a fraction of pixels in x.

    :returns: (y_shifts, x_shifts) Two arrays (num_frames) with the y, x motion shifts
    :rtype: (np.array, np.array)
    """
    # Add third dimension if scan is a single image
    if scan.ndim == 2:
        scan = np.expand_dims(scan, -1)

    # Get some params
    image_height, image_width, num_frames = scan.shape
    max_y_shift = round(image_height * max_fraction_yshift)  # max num_pixels to move in y
    max_x_shift = round(image_width * max_fraction_xshift)  # max num_pixels to move in x
    skip_rows = round(image_height * 0.10)  # rows near the top or bottom have artifacts
    skip_cols = round(image_width * 0.10)  # so do columns

    # Discard some rows/cols
    template = template[skip_rows: -skip_rows, skip_cols: -skip_cols]
    scan = scan[skip_rows: -skip_rows, skip_cols: -skip_cols, :]

    # Get fourier transform of template
    template_freq = np.fft.fftn(template)

    # Compute subpixel shifts per image (verbatim from skimage.feature.register_translation)
    y_shifts = np.empty(num_frames)
    x_shifts = np.empty(num_frames)
    for i in range(num_frames):
        image_freq = np.fft.fftn(scan[:, :, i])
        # yx_shift = feature.register_translation(image_freq, template_freq, 10,
        #                                        space='fourier')[0]

        # To avoid the overhead of register_translation (which computes a set of other
        # things on top of the motion correction shifts), we copy only the relevant code
        # from skimage.feature.register_translation
        src_freq = image_freq
        target_freq = template_freq
        upsample_factor = 10  # motion correction to a tenth of a pixel

        #########################################################################
        from skimage.feature.register_translation import _upsampled_dft

        # Whole-pixel shift - Compute cross-correlation by an IFFT
        shape = src_freq.shape
        image_product = src_freq * target_freq.conj()
        cross_correlation = np.fft.ifftn(image_product)



        #cc_image = np.fft.fftshift(cross_correlation)
        #return cross_correlation
        #import matplotlib.pyplot as plt
        #print(cross_correlation.shape)
        #print(cross_correlation.dtype)
        #fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
        #axes[0].imshow(scan[:, :, i])
        #axes[1].imshow(template)
        #axes[2].imshow(np.abs(cc_image))
        #plt.show()






        # Locate maximum
        maxima = np.unravel_index(np.argmax(np.abs(cross_correlation)),
                                  cross_correlation.shape)
        midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])

        shifts = np.array(maxima, dtype=np.float64)
        shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

        # Initial shift estimate in upsampled grid
        shifts = np.round(shifts * upsample_factor) / upsample_factor
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        # Center of output array at dftshift + 1
        dftshift = np.fix(upsampled_region_size / 2.0)
        upsample_factor = np.array(upsample_factor, dtype=np.float64)
        normalization = (src_freq.size * upsample_factor ** 2)
        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts * upsample_factor
        cross_correlation = _upsampled_dft(image_product.conj(), upsampled_region_size,
                                           upsample_factor, sample_region_offset).conj()
        cross_correlation /= normalization
        # Locate maximum and map back to original pixel grid
        maxima = np.array(np.unravel_index(
            np.argmax(np.abs(cross_correlation)),
            cross_correlation.shape),
            dtype=np.float64)
        maxima -= dftshift
        shifts = shifts + maxima / upsample_factor
        ################################################################################
        yx_shift = shifts

        y_shifts[i] = yx_shift[0]
        x_shifts[i] = yx_shift[1]

    # Limit max shift
    #extreme_y_shifts = abs(y_shifts) > max_y_shift
    #extreme_x_shifts = abs(x_shifts) > max_x_shift
    #y_shifts[extreme_y_shifts] = (np.sign(y_shifts) * max_y_shift)[extreme_y_shifts]
    #x_shifts[extreme_x_shifts] = (np.sign(x_shifts) * max_x_shift)[extreme_x_shifts]

    return y_shifts, x_shifts

#***************************************************************************************
def compute_raster_phase(image, temporal_fill_fraction):
    """ Compute raster correction for bidirectional resonant scanners.

    It shifts the even and odd rows of the image in the x axis to find the scan angle
    that aligns them better. Positive raster phase will shift even rows to the right and
    odd rows to the left (assuming first row is row 0).

    :param np.array image: The image to be corrected.
    :param float temporal_fill_fraction: Fraction of time during which the scan is
        recording a line against the total time per line.

    :return: An angle (in radians). Estimate of the mismatch angle between the expected
         initial angle and the one recorded.
    :rtype: float
    """
    # Make sure image has even number of rows (so number of even and odd rows is the same)
    image = image[:-1] if image.shape[0] % 2 == 1 else image

    # Get some params
    image_height, image_width = image.shape
    skip_rows = round(image_height * 0.05) # rows near the top or bottom have artifacts
    skip_cols = round(image_width * 0.10) # so do columns

    # Create images with even and odd rows
    even_rows = image[::2][skip_rows: -skip_rows]
    odd_rows = image[1::2][skip_rows: -skip_rows]

    # Scan angle at which each pixel was recorded.
    max_angle = (np.pi / 2) * temporal_fill_fraction
    scan_angles = np.linspace(-max_angle, max_angle, image_width + 2)[1:-1]
    #sin_index = np.sin(scan_angles)

    # Greedy search for the best raster phase: starts at coarse estimates and refines them
    even_interp = interp.interp1d(scan_angles, even_rows, fill_value='extrapolate')
    odd_interp = interp.interp1d(scan_angles, odd_rows, fill_value='extrapolate')
    angle_shift = 0
    for scale in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        angle_shifts = angle_shift + scale * np.linspace(-9, 9, 19)
        match_values = []
        for new_angle_shift in angle_shifts:
            shifted_evens = even_interp(scan_angles + new_angle_shift)
            shifted_odds = odd_interp(scan_angles - new_angle_shift)
            match_values.append(np.sum(shifted_evens[:, skip_cols: -skip_cols] *
                                       shifted_odds[:, skip_cols: -skip_cols]))
        angle_shift = angle_shifts[np.argmax(match_values)]

    return angle_shift


def compute_motion_shifts(scan, template, fix_outliers=True, outlier_threshold=0.05,
                          smooth_shifts=True, smoothing_window_size=5):
    """ Compute shifts in x and y for rigid subpixel motion correction.

    Returns the number of pixels that each image in the scan was to the right (x_shift)
    or below (y_shift) the template. Negative shifts mean the image was to the left or
    above the template.

    :param np.array template: 2-d template image. Each frame in scan is aligned to this.
    :param np.array scan: 2 or 3-dimensional scan (image_height, image_width[, num_frames]).
    :param bool fix_outliers: If True, look for spikes in motion shifts and sets them to
        the mean around them.
    :param float outlier_threshold: Threshold (as a fraction of dimension length) for
        outlier detection.
    :param bool smooth_shifts: If True, smooth the timeseries of shifts.
    :param int smoothing_window_size: Size of the Hann window (pixels) for smoothing.

    :returns: (y_shifts, x_shifts) Two arrays (num_frames) with the y, x motion shifts
    :returns: (y_outliers, x_outliers) Two boolean arrays (num_frames) with True for y, x
        outliers
    """
    # Add third dimension if scan is a single image
    if scan.ndim == 2:
        scan = np.expand_dims(scan, -1)

    # Get some params
    image_height, image_width, num_frames = scan.shape
    max_y_shift = round(image_height * outlier_threshold)
    max_x_shift = round(image_width * outlier_threshold)

    # Get fourier transform of template
    template_freq = np.fft.fftn(template)

    # Compute subpixel shifts per image (verbatim from skimage.feature.register_translation)
    y_shifts = np.empty(num_frames)
    x_shifts = np.empty(num_frames)
    for i in range(num_frames):
        image_freq = np.fft.fftn(scan[:, :, i])
        # yx_shift = feature.register_translation(image_freq, template_freq, 10,
        #                                        space='fourier')[0]

        # To avoid the overhead of register_translation (which computes a set of other
        # things on top of the motion correction shifts), we copy only the relevant code
        # from skimage.feature.register_translation
        src_freq = image_freq
        target_freq = template_freq
        upsample_factor = 10 # motion correction to a tenth of a pixel

        #########################################################################
        from skimage.feature.register_translation import _upsampled_dft

        # Whole-pixel shift - Compute cross-correlation by an IFFT
        shape = src_freq.shape
        image_product = src_freq * target_freq.conj()
        cross_correlation = np.fft.ifftn(image_product)

        # Locate maximum
        maxima = np.unravel_index(np.argmax(np.abs(cross_correlation)),
                                  cross_correlation.shape)
        midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])

        shifts = np.array(maxima, dtype=np.float64)
        shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

        # Initial shift estimate in upsampled grid
        shifts = np.round(shifts * upsample_factor) / upsample_factor
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        # Center of output array at dftshift + 1
        dftshift = np.fix(upsampled_region_size / 2.0)
        upsample_factor = np.array(upsample_factor, dtype=np.float64)
        normalization = (src_freq.size * upsample_factor ** 2)
        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts * upsample_factor
        cross_correlation = _upsampled_dft(image_product.conj(), upsampled_region_size,
                                           upsample_factor, sample_region_offset).conj()
        cross_correlation /= normalization
        # Locate maximum and map back to original pixel grid
        maxima = np.array(np.unravel_index(
            np.argmax(np.abs(cross_correlation)),
            cross_correlation.shape),
            dtype=np.float64)
        maxima -= dftshift
        shifts = shifts + maxima / upsample_factor
        ################################################################################
        yx_shift = shifts

        y_shifts[i] = yx_shift[0]
        x_shifts[i] = yx_shift[1]


    # Detect outliers and set their value to the mean around them.
    y_outliers = None
    x_outliers = None
    if fix_outliers:
        y_shifts, y_outliers = _fix_outliers(y_shifts, max_y_shift)
        x_shifts, x_outliers = _fix_outliers(x_shifts, max_x_shift)

    # Smooth the shifts temporally
    if smooth_shifts:
        smoothing_window = signal.hann(smoothing_window_size) + 0.05 # 0.05 raises it so edges are not zero.
        norm_smoothing_window = smoothing_window / sum(smoothing_window)
        y_shifts = signal.convolve(y_shifts, norm_smoothing_window, mode='same')
        x_shifts = signal.convolve(x_shifts, norm_smoothing_window, mode='same')

    return y_shifts, x_shifts, y_outliers, x_outliers

def _fix_outliers(shifts, max_shift):
    """ Sets shifts whose deviation from the moving average is greater than max_shift to
    the moving average value at that position.

    ..note:: Changes done in place.
    """
    moving_average = np.convolve(shifts, np.ones(100) / 100, 'same')
    extreme_shifts = abs(shifts - moving_average) > max_shift
    shifts[extreme_shifts] = moving_average[extreme_shifts]
    return shifts, extreme_shifts


def correct_raster(scan, raster_phase, temporal_fill_fraction, in_place=True):
    """ Raster correction for resonant scans.

    Corrects multi-photon images in n-dimensional scans. Positive raster phase shifts
    even lines to the left and odd lines to the right. Negative raster phase shifts even
    lines to the right and odd lines to the left.

    :param np.array scan: Volume with images to be corrected in the first two dimensions.
        Works for 2-dimensions and up, usually (image_height, image_width, num_frames).
    :param float raster_phase: Angle difference between expected and recorded scan angle.
    :param float temporal_fill_fraction: Ratio between active acquisition and total
        length of the scan line.
    :param bool in_place: If True (default), the original array is modified in place.

    :return: Raster-corrected scan.
    :rtype: Same as scan if scan.dtype is subtype of np.float, else np.float32.

    :raises: PipelineException
    """
    # Basic checks
    if not isinstance(scan, np.ndarray):
        raise PipelineException('Scan needs to be a numpy array.')
    if scan.ndim < 2:
        raise PipelineException('Scan with less than 2 dimensions.')

    # Assert scan is float
    if not np.issubdtype(scan.dtype, np.float):
         print('Warning: Changing scan type from', str(scan.dtype), 'to np.float32')
         scan = scan.astype(np.float32, copy=(not in_place))
    elif not in_place:
         scan = scan.copy() # copy it anyway preserving the original float dtype

    # Get some dimensions
    original_shape = scan.shape
    image_height = original_shape[0]
    image_width = original_shape[1]

    # Scan angle at which each pixel was recorded.
    max_angle = (np.pi / 2) * temporal_fill_fraction
    scan_angles = np.linspace(-max_angle, max_angle, image_width + 2)[1:-1]

    # We iterate over every image in the scan (first 2 dimensions). Same correction
    # regardless of what channel, slice or frame they belong to.
    reshaped_scan = np.reshape(scan, (image_height, image_width, -1))
    num_images = reshaped_scan.shape[-1]
    for i in range(num_images):
        # Get current image
        image = reshaped_scan[:, :, i]

        # Correct even rows of the image (0, 2, ...)
        interp_function = interp.interp1d(scan_angles, image[::2, :], bounds_error=False,
                                          fill_value='extrapolate', copy=False)
        reshaped_scan[::2, :, i] = interp_function(scan_angles + raster_phase)

        # Correct odd rows of the image (1, 3, ...)
        interp_function = interp.interp1d(scan_angles, image[1::2, :], bounds_error=False,
                                          fill_value='extrapolate', copy=False)
        reshaped_scan[1::2, :, i] = interp_function(scan_angles - raster_phase)

    scan = np.reshape(reshaped_scan, original_shape)
    return scan


def correct_motion(scan, xy_shifts, in_place=True):
    """ Motion correction for multi-photon scans.

    Shifts each image in the scan x_shift pixels to the left and y_shift pixels up.

    :param np.array scan: Volume with images to be corrected in the first two dimensions.
        Works for 2-dimensions and up, usually (image_height, image_width, num_frames).
    :param list/np.array xy_shifts: Volume with x, y motion shifts for each image in the
        first dimension: usually (2 x num_frames).
    :param bool in_place: If True (default), the original array is modified in place.

    :return: Motion corrected scan
    :rtype: Same as scan if scan.dtype is subtype of np.float, else np.float32.

    :raises: PipelineException
    """
    # Basic checks
    if not isinstance(scan, np.ndarray):
        raise PipelineException('Scan needs to be a numpy array.')
    if scan.ndim < 2:
        raise PipelineException('Scan with less than 2 dimensions.')

    # Assert scan is float (integer precision is not good enough)
    if not np.issubdtype(scan.dtype, np.float):
        print('Warning: Changing scan type from', str(scan.dtype), 'to np.float32')
        scan = scan.astype(np.float32, copy=(not in_place))
    elif not in_place:
        scan = scan.copy() # copy it anyway preserving the original dtype

    # Get some dimensions
    original_shape = scan.shape
    image_height = original_shape[0]
    image_width = original_shape[1]

    # Reshape input (to deal with more than 2-D volumes)
    reshaped_scan = np.reshape(scan, (image_height, image_width, -1))
    num_images = reshaped_scan.shape[-1]
    reshaped_xy = np.reshape(xy_shifts, (2, -1))

    if reshaped_xy.shape[-1] != reshaped_scan.shape[-1]:
        raise PipelineException('Scan and motion arrays have different dimensions')

    # Over every image (as long as the x and y shift is defined)
    for i in range(num_images):
        [x_shift, y_shift] = reshaped_xy[:, i]
        if not np.isnan(x_shift) and not np.isnan(y_shift):

            # Get current image
            image = reshaped_scan[:, :, i]

            # Create interpolation function
            interp_function = interp.interp2d(range(image_width), range(image_height),
                                              image, kind='cubic', copy=False)

            # Evaluate on the original image plus offsets
            reshaped_scan[:, :, i] = interp_function(np.arange(image_width) + x_shift,
                                                     np.arange(image_height) + y_shift)

    scan = np.reshape(reshaped_scan, original_shape)
    return scan