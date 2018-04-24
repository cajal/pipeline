""" Utilities for motion and raster correction of resonant scans. """
import numpy as np
from scipy import interpolate  as interp
from scipy import signal
from scipy import ndimage

from ..exceptions import PipelineException
from ..utils.signal import mirrconv

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


def compute_motion_shifts(scan, template, in_place=True, num_threads=8):
    """ Compute shifts in y and x for rigid subpixel motion correction.

    Returns the number of pixels that each image in the scan was to the right (x_shift)
    or below (y_shift) the template. Negative shifts mean the image was to the left or
    above the template.

    :param np.array scan: 2 or 3-dimensional scan (image_height, image_width[, num_frames]).
    :param np.array template: 2-d template image. Each frame in scan is aligned to this.
    :param bool in_place: Whether the scan can be overwritten.
    :param int num_threads: Number of threads used for the ffts.

    :returns: (y_shifts, x_shifts) Two arrays (num_frames) with the y, x motion shifts.

    ..note:: Based in imreg_dft.translation().
    """
    import pyfftw
    from imreg_dft import utils

    # Add third dimension if scan is a single image
    if scan.ndim == 2:
        scan = np.expand_dims(scan, -1)

    # Get some params
    image_height, image_width, num_frames = scan.shape
    taper = np.outer(signal.tukey(image_height, 0.2), signal.tukey(image_width, 0.2))

    # Prepare fftw
    frame = pyfftw.empty_aligned((image_height, image_width), dtype='complex64')
    fft = pyfftw.builders.fft2(frame, threads=num_threads, overwrite_input=in_place,
                               avoid_copy=True)
    ifft = pyfftw.builders.ifft2(frame, threads=num_threads, overwrite_input=in_place,
                                 avoid_copy=True)

    # Get fourier transform of template
    template_freq = fft(template * taper).conj() # we only need the conjugate
    abs_template_freq = abs(template_freq)
    eps = abs_template_freq.max() * 1e-15

    # Compute subpixel shifts per image
    y_shifts = np.empty(num_frames)
    x_shifts = np.empty(num_frames)
    for i in range(num_frames):
        # Compute correlation via cross power spectrum
        image_freq = fft(scan[:, :, i] * taper)
        cross_power = (image_freq * template_freq) / (abs(image_freq) * abs_template_freq + eps)
        shifted_cross_power = np.fft.fftshift(abs(ifft(cross_power)))

        # Get best shift
        shifts = np.unravel_index(np.argmax(shifted_cross_power), shifted_cross_power.shape)
        shifts = utils._interpolate(shifted_cross_power, shifts, rad=3)

        # Map back to deviations from center
        y_shifts[i] = shifts[0] - image_height // 2
        x_shifts[i] = shifts[1] - image_width // 2

    return y_shifts, x_shifts


def fix_outliers(y_shifts, x_shifts, max_y_shift=20, max_x_shift=20, method='median'):
    """ Look for spikes in motion shifts and set them to a sensible value.

    Reject any shift whose y or x shift is higher than max_y_shift/max_x_shift pixels
    from the median/linear estimate/moving average. Outliers filled by interpolating
    valid points; in the edges filled with the median/linear estimate/moving average.

    :param np.array y_shifts/x_shifts: Shifts in y, x.
    :param float max_y_shift/max_x_shifts: Number of pixels used as threshold to classify
        a point as an outlier in y, x.
    :param string method: One of 'mean' or 'trend'.
        'median': Detect outliers as deviations from the median of the shifts.
        'linear': Detect outliers as deviations from a line estimated from the shifts.
        'trend': Detect outliers as deviations from the shift trend computed as a moving
            average over the entire scan.

    :returns: (y_shifts, x_shifts) Two arrays (num_frames) with the fixed motion shifts.
    :returns: (outliers) A boolean array (num_frames) with True for outlier frames.
    """
    # Basic checks
    num_frames = len(y_shifts)
    if num_frames < 5:
        return y_shifts, x_shifts, np.full(num_frames, False)

    # Copy shifts to avoid changing originals
    y_shifts, x_shifts = y_shifts.copy(), x_shifts.copy()

    # Detrend shifts
    if method == 'median':
        y_trend = np.median(y_shifts)
        x_trend = np.median(x_shifts)
    elif method == 'linear':
        x_trend = _fit_robust_line(x_shifts)
        y_trend = _fit_robust_line(y_shifts)
    else: # trend
        window_size = min(101, num_frames)
        window_size -= 1 if window_size % 2 == 0 else 0
        y_trend = mirrconv(y_shifts, np.ones(window_size) / window_size)
        x_trend = mirrconv(x_shifts, np.ones(window_size) / window_size)

    # Subtract trend from shifts
    y_shifts -= y_trend
    x_shifts -= x_trend

    # Get outliers
    outliers = np.logical_or(abs(y_shifts) > max_y_shift, abs(x_shifts) > max_x_shift)

    # Interpolate outliers
    num_outliers = np.sum(outliers)
    if num_outliers < num_frames - 1: # at least two good points needed for interpolation
        #indices = np.arange(len(x_shifts))
        #y_shifts = np.interp(indices, indices[~outliers], y_shifts[~outliers], left=0, right=0)
        #x_shifts = np.interp(indices, indices[~outliers], x_shifts[~outliers], left=0, right=0)
        y_shifts[outliers] = 0
        x_shifts[outliers] = 0
    else:
        print('Warning: {} out of {} frames were outliers.'.format(num_outliers, num_frames))
        y_shifts = 0
        x_shifts = 0

    # Add trend back to shifts
    y_shifts += y_trend
    x_shifts += x_trend

    return y_shifts, x_shifts, outliers


def _fit_robust_line(shifts):
    """ Use a robust linear regression algorithm to fit a line to the data."""
    from sklearn.linear_model import TheilSenRegressor

    X = np.arange(len(shifts)).reshape(-1, 1)
    y = shifts
    model = TheilSenRegressor() # robust regression
    model.fit(X, y)
    line = model.predict(X)

    return line


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
    if not np.issubdtype(scan.dtype, np.floating):
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
                                          fill_value=0, copy=(not in_place))
        reshaped_scan[::2, :, i] = interp_function(scan_angles + raster_phase)

        # Correct odd rows of the image (1, 3, ...)
        interp_function = interp.interp1d(scan_angles, image[1::2, :], bounds_error=False,
                                          fill_value=0, copy=(not in_place))
        reshaped_scan[1::2, :, i] = interp_function(scan_angles - raster_phase)

    scan = np.reshape(reshaped_scan, original_shape)
    return scan


def correct_motion(scan, x_shifts, y_shifts, in_place=True):
    """ Motion correction for multi-photon scans.

    Shifts each image in the scan x_shift pixels to the left and y_shift pixels up.

    :param np.array scan: Volume with images to be corrected in the first two dimensions.
        Works for 2-dimensions and up, usually (image_height, image_width, num_frames).
    :param list/np.array x_shifts: 1-d array with x motion shifts for each image.
    :param list/np.array y_shifts: 1-d array with x motion shifts for each image.
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
    if np.ndim(y_shifts) != 1 or np.ndim(x_shifts) != 1:
        raise PipelineException('Dimension of one or both motion arrays differs from 1.')
    if len(x_shifts) != len(y_shifts):
        raise PipelineException('Length of motion arrays differ.')

    # Assert scan is float (integer precision is not good enough)
    if not np.issubdtype(scan.dtype, np.floating):
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
    if reshaped_scan.shape[-1] != len(x_shifts):
        raise PipelineException('Scan and motion arrays have different dimensions')

    # Ignore NaN values (present in some older data)
    y_clean, x_clean = y_shifts.copy(), x_shifts.copy()
    y_clean[np.logical_or(np.isnan(y_shifts), np.isnan(x_shifts))] = 0
    x_clean[np.logical_or(np.isnan(y_shifts), np.isnan(x_shifts))] = 0

    # Shift each frame
    for i, (y_shift, x_shift) in enumerate(zip(y_clean, x_clean)):
        image = reshaped_scan[:, :, i].copy()
        ndimage.interpolation.shift(image, (-y_shift, -x_shift), order=1,
                                    output=reshaped_scan[:, :, i])

    scan = np.reshape(reshaped_scan, original_shape)
    return scan