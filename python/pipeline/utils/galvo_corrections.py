from .. import PipelineException
from scipy import interpolate  as interp
from scipy import signal
import numpy as np

def compute_motion_shifts(field, template, smooth_shifts=True, smoothing_window_size=5,
                          max_fraction_yshift=0.1, max_fraction_xshift=0.1):
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
    
    :returns: (y_shifts, x_shifts) Each is an array ([num_frames]) with the y, x motion 
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
    max_y_shift = int(round(image_height * max_fraction_yshift))  # max num_pixels to move in y
    max_x_shift = int(round(image_width * max_fraction_xshift))

    # Cut center patch of frames
    frame_centers = field[max_y_shift: -max_y_shift, max_x_shift: -max_x_shift, :]

    y_shifts = np.zeros(num_frames)
    x_shifts = np.zeros(num_frames)
    for i in range(num_frames):

        # Convolve frame center on template and find highest value.
        conv_image = signal.correlate(template, frame_centers[:, :, i], mode='valid')
        y_shift, x_shift = np.unravel_index(np.argmax(conv_image), conv_image.shape)
        y_shifts[i] = max_y_shift - y_shift
        x_shifts[i] = max_x_shift - x_shift

        # if align_subpixels:
        #     """Transcribed from commons.ne7.ip.measureShift.m. Doesn't quite work."""
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
        #     quarter_height = int(round(image_height / 4))
        #     quarter_width = int(round(image_width / 4))
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


def compute_raster_phase(image):
    """"""
    print('Warning: Raster correction not implemented yet.')
    #TODO: Compute raster correction
    return 0

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
    if not np.issubdtype(scan.dtype, np.float):
        print('Changing scan type from', str(scan.dtype), 'to double')
        scan = np.double(scan)

    # Get some dimensions
    original_shape = scan.shape
    image_height = original_shape[0]
    image_width = original_shape[1]

    # If copy is needed
    if not in_place:
        scan = scan.copy()

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


def correct_raster(scan, raster_phase, fill_fraction, in_place=True):
    """ Raster correction for resonant scanners.

    Corrects two-photon images in n-dimensional scans, usual shape is [image_height,
    image_width, channels, slices, num_frames]. Works for 2-d arrays and up (assuming
    images are in the first two dimensions). Based on Matlab implementation of
    ne7.ip.correctRaster()
    
    Positive raster phase shifts even lines to the left and odd lines to the right. 
    Negative raster phase shifts even lines to the right and odd lines to the left.  

    :param np.array scan: Volume with images to be corrected in the first two dimensions.
    :param float raster_phase: Phase difference between odd and even lines.
    :param float fill_fraction: Ratio between active acquisition and total length of the
    scan line.
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
    if not np.issubdtype(scan.dtype, np.float):
        print('Changing scan type from', str(scan.dtype), 'to double')
        scan = np.double(scan)

    # Get some dimensions
    original_shape = scan.shape
    image_height = original_shape[0]
    image_width = original_shape[1]
    half_width = image_width / 2

    # If copy is needed
    if not in_place:
        scan = scan.copy()

    # Create interpolation points for sinusoidal raster
    index  = np.linspace(-half_width + 0.5, half_width - 0.5, image_width) / half_width
    time_index = np.arcsin(index * fill_fraction)
    interp_points_even = np.sin(time_index + raster_phase) / fill_fraction
    interp_points_odd = np.sin(time_index - raster_phase) / fill_fraction

    # We iterate over every image in the scan (first 2 dimensions). Same correction
    # regardless of what channel, slice or frame they belong to.
    reshaped_scan = np.reshape(scan, (image_height, image_width, -1))
    num_images = reshaped_scan.shape[-1]
    for i in range(num_images):
        # Get current image
        image = reshaped_scan[:, :, i]
        mean_intensity = np.mean(image)  # to fill outside the interpolation range

        # Correct even rows of the image (0, 2, ...)
        interp_function = interp.interp1d(index, image[::2, :], fill_value=mean_intensity,
                                          bounds_error=False, copy=False)
        reshaped_scan[::2, :, i] = interp_function(interp_points_even)

        # Correct odd rows of the image (1, 3, ...)
        interp_function = interp.interp1d(index, image[1::2, :], bounds_error=False,
                                          fill_value=mean_intensity, copy=False)
        reshaped_scan[1::2, :, i] = interp_function(interp_points_odd)

    scan = np.reshape(reshaped_scan, original_shape)
    return scan