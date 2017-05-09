from .. import PipelineException
from scipy import interpolate  as interp
from scipy import signal
import numpy as np

def compute_motion_shifts(field, template, smooth_shifts=True, smoothing_window_size=5,
                          max_fraction_yshift=0.10, max_fraction_xshift=0.10):
    """ Compute shifts in x and y for rigid motion correction.
    
    A patch is taken from the center of each frame (the margin size that is not included
    limits the maximum possible pixel shift in that dimension) and convolved with the 
    template to get the amount of shifting needed to align the center of the frame with 
    the template: 0 if center of frame = center of template. This is an approximation of 
    the full 2-d convolution: works well if center patch is big and has the advantage of 
    choosing a good shift inside the allowed limit and not having a preference for zero
    shifts.
    
    A positive x_shift means that the image needs to be moved x_shift pixels left, a 
    positive y_shift means the image needs to be moved y_shift pixels up.
    
    :param np.array template: 2-d template image. Each frame in field is aligned to this.
    :param np.array field: 2-d or 3-d array with images in the first two dimension,
        usually [image_height, image_width, num_frames]
    :param bool smooth_shifts: If True, smooth the timeseries of shifts.
    :param int smoothing_window_size: Size of the Hann window used for smoothing.
    :param max_fraction_yshift: Maximum allowed motion in y as a fraction of pixels in y.
    :param max_fraction_xshift: Maximum allowed motion in x as a fraction of pixels in x.
    
    :return: (y_shifts, x_shifts) Each is an array ([num_frames]) with the y, x motion 
        shifts needed to align each frame with the template.
    :rtype: (np.array, np.array) 
    """
    # Add third dimension if field is a single image
    if field.ndim == 2:
        field = np.expand_dims(field, -1)

    # Assert template is float. signal.correlate needs one of its arguments to be float
    if not np.issubdtype(template.dtype, np.float):
        template = template.astype(np.float32)

    # Get some params
    image_height, image_width, num_frames = field.shape
    max_y_shift = round(image_height * max_fraction_yshift)  # max num_pixels to move in y
    max_x_shift = round(image_width * max_fraction_xshift)
    skip_rows = round(image_height * 0.10)  # rows near the top or bottom have artifacts
    skip_cols = round(image_width * 0.10)  # so do columns

    # Discard some rows/cols and cut center patch of frames
    template = template[skip_rows: -skip_rows, skip_cols: -skip_cols]
    frame_centers = field[skip_rows + max_y_shift: -skip_rows - max_y_shift,
                    skip_cols + max_x_shift: -skip_cols - max_x_shift, :]

    y_shifts = np.zeros(num_frames)
    x_shifts = np.zeros(num_frames)
    for i in range(num_frames):

        # Convolve frame center on template and find highest value.
        conv_image = signal.correlate(template, frame_centers[:, :, i], mode='valid')
        y_shift, x_shift = np.unravel_index(np.argmax(conv_image), conv_image.shape)
        y_shifts[i] = max_y_shift - y_shift
        x_shifts[i] = max_x_shift - x_shift

        # if align_subpixels:
        #     """Transcribed from commons.ne7.ip.measureShift.m."""
        #     from scipy import fftpack
        #     fxcorr = fftpack.fft2(field[:, :, i]) * np.conj(fftpack.fft2(template))
        #
        #     # Shift by the whole number of pixels
        #     half_height = np.floor(image_height / 2)
        #     half_width = np.floor(image_width / 2)
        #     y_freqs = np.arange(-half_height, half_height + image_height % 2) / image_height
        #     x_freqs = np.arange(-half_width, half_width + image_width % 2) / image_width
        #     shifts_in_freq = np.outer(np.exp(2 * np.pi * y_shifts[i] * 1j * y_freqs),
        #                               np.exp(2 * np.pi * x_shifts[i] * 1j * x_freqs))
        #     shifted_fxcorr = fxcorr * fftpack.fftshift(shifts_in_freq)
        #
        #     # Get phases and magnitudes from fxcorr
        #     phases = fftpack.fftshift(np.angle(shifted_fxcorr))
        #     magnitudes = fftpack.fftshift(np.abs(shifted_fxcorr))
        #     y_weighted_magnitudes = (magnitudes.T * y_freqs).T
        #     x_weighted_magnitudes = magnitudes * x_freqs
        #     phase_times_magnitude = phases * magnitudes
        #
        #     # Select only those with low frequencies (half Nyquist, freqs < 0.25)
        #     quarter_height = round(image_height / 4)
        #     quarter_width = round(image_width / 4)
        #     y_mag = y_weighted_magnitudes[quarter_height: -quarter_height, quarter_width: -quarter_width].ravel()
        #     x_mag = x_weighted_magnitudes[quarter_height: -quarter_height, quarter_width: -quarter_width].ravel()
        #     phase = phase_times_magnitude[quarter_height: -quarter_height, quarter_width: -quarter_width].ravel()
        #
        #     # To find the slope in y and x fit:
        #     # phase_times_magnitude = y_shift * y_weighted_magnitudes
        #     #                         + x_shift * x_weighted_magnitudes
        #     A = np.stack([y_mag, x_mag], axis=1)
        #     b = phase
        #     y_slope, x_slope = np.linalg.lstsq(A, b)[0]
        #
        #     # Apply subpixel shifts
        #     y_shifts[i] -= (y_slope / (2 * np.pi))
        #     x_shifts[i] -= (x_slope / (2 * np.pi))

    if smooth_shifts:
        # Temporal smoothing of the y and x shifts
        smoothing_window = signal.hann(smoothing_window_size)
        norm_smoothing_window = smoothing_window / sum(smoothing_window)
        y_shifts = signal.convolve(y_shifts, norm_smoothing_window, mode='same')
        x_shifts = signal.convolve(x_shifts, norm_smoothing_window, mode='same')

    return y_shifts, x_shifts


def compute_raster_phase(image, temporal_fill_fraction):
    """ Compute raster correction for bidirectional resonant scanners.
    
    It shifts the odd and even rows to the left and right by an increasing angle to find
    the best match. Positive raster phase will shift even rows to the right and odd rows 
    to the left.
    
    :param np.array image: The image to be corrected.
    :param float temporal_fill_fraction: Fraction of time during which the scan is 
        recording a line against the total time per line. 
        
    :return: An angle (in radians). Estimate of the mismatch angle between the expected 
        initial angle and the one recorded.
    :rtype: float
    """
    # Make sure image has even number of rows (so # even and odd rows is the same)
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

    # Old version
    # Create time index
    # half_width = image_width / 2
    # time_range = ((np.arange(image_width) + 0.5) - half_width) / half_width # sin(angle)
    # time_range = time_range * temporal_fill_fraction # scan is off on the edges,
    # This houdl have been temporal
    # scan_angles = np.arcsin(time_range)

    # Iteratively find the best raster phase, by shifting odd and even rows.
    # raster_phase = 0
    # step = 0.02
    # phase_shifts = np.array([-0.5, -0.25, -0.1, 0.1, 0.25, 0.5])
    # even_interp = interp.interp1d(sin_index, even_rows)
    # odd_interp = interp.interp1d(sin_index, odd_rows)
    # while step > 1e-4:
    #     phases = raster_phase + (step * phase_shifts) * spatial_fill_fraction
    #     match = []
    #     for new_phase in phases:
    #         shifted_evens = even_interp(np.sin(scan_angles + new_phase) / spatial_fill_fraction)
    #         shifted_odds = odd_interp(np.sin(scan_angles - new_phase) / spatial_fill_fraction)
    #         match.append(np.sum(shifted_evens[:, 10:-10] * shifted_odds[:, 10:-10]))
    #
    #     A = np.stack([phases**2, phases, np.ones_like(phases)])
    #     b = np.array(match)
    #     alpha_2, alpha_1, alpha_0 = np.linalg.lstsq(A, b)[0]
    #     raster_phase = max(alpha_2, min(phases[-1], -(alpha_1/alpha_2) / 2))
    #     step = step/4
    return angle_shift

def correct_motion(scan, xy_shifts, in_place=True):
    """ Motion correction for two-photon scans.

    Shifts each image in the scan x_shift pixels to the left and y_shift pixels up. Works 
    for 2-d arrays and up.

    :param np.array scan: Volume with images to be corrected in the first two dimensions.
    :param list/np.array xy_shifts: Volume with x, y motion shifts for each image in the
    first dimension: usually [2 x num_frames].
    :param bool in_place: If True (default), the original array is modified in memory.

    :return: Motion corrected scan
    :rtype: Same as scan if scan.dtype is subtype of np.float, else np.double.

    :raises: PipelineException
    """
    # Basic checks
    if not isinstance(scan, np.ndarray):
        raise PipelineException('Scan needs to be a numpy array.')
    if scan.ndim < 2:
        raise PipelineException('Scan with less than 2 dimensions.')

    # Assert scan is float (integer precision is not good enough)
    if not np.issubdtype(scan.dtype, np.float):
        print('Changing scan type from', str(scan.dtype), 'to double')
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


def correct_raster(scan, raster_phase, temporal_fill_fraction, in_place=True):
    """ Raster correction for resonant scanners.

    Corrects two-photon images in n-dimensional scans, usual shape is [image_height,
    image_width, channels, slices, num_frames]. Works for 2-d arrays and up (assuming
    images are in the first two dimensions). Based on Matlab implementation of
    ne7.ip.correctRaster()
    
    Positive raster phase shifts even lines to the left and odd lines to the right. 
    Negative raster phase shifts even lines to the right and odd lines to the left.  

    :param np.array scan: Volume with images to be corrected in the first two dimensions.
    :param float raster_phase: Phase difference between odd and even lines.
    :param float temporal_fill_fraction: Ratio between active acquisition and total length
        of the scan line.
    :param bool in_place: If True (default), the original array is modified in memory.

    :return: Raster-corrected scan.
    :rtype: Same as scan if scan.dtype is subtype of np.float, else np.double.

    :raises: PipelineException
    """
    # Basic checks
    if not isinstance(scan, np.ndarray):
        raise PipelineException('Scan needs to be a numpy array.')
    if scan.ndim < 2:
        raise PipelineException('Scan with less than 2 dimensions.')

    # Assert scan is float
    if not np.issubdtype(scan.dtype, np.float):
        print('Changing scan type from', str(scan.dtype), 'to float')
        scan = scan.astype(np.float32, copy=(not in_place))
    elif not in_place:
        scan = scan.copy() # copy it anyway preserving the original dtype

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