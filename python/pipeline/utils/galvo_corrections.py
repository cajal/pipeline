""" Utilities for motion and raster correction of resonant scans. """
import numpy as np
import datajoint as dj
from scipy import interpolate  as interp
from scipy import signal
from scipy import ndimage
from tqdm import tqdm

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

def _get_field_size_info(key):
    """
    Small utility function to fetch field height/width in pixels.
    Required to make template functions below pipeline agnostic.
    
    Returns:
        tuple: (height, width, height_microns_per_pixel, width_microns_per_pixel)
    """
    # Define virtual modules here to prevent circular imports
    reso = dj.create_virtual_module("reso", "pipeline_reso")
    meso = dj.create_virtual_module("meso", "pipeline_meso")

    # Fetch relevant field size info
    if len(reso.ScanInfo & key) > 0:
        px_height, px_width, um_height, um_width = (reso.ScanInfo & key).fetch1(
            "px_height", "px_width", "um_height", "um_width"
        )
    elif len(meso.ScanInfo & key) > 0:
        px_height, px_width, um_height, um_width = (meso.ScanInfo.Field & key).fetch1(
            "px_height", "px_width", "um_height", "um_width"
        )
    else:
        raise PipelineException(f"Could not find scan info for key: {key}.")

    return px_height, px_width, um_height / px_height, um_width / px_width

def _get_pipe(key):
    """
    Small utility function to return instance of meso/reso pipeline.
    Required to make template functions below pipeline agnostic.
    """
    # Define virtual modules here to prevent circular imports
    reso = dj.create_virtual_module("reso", "pipeline_reso")
    meso = dj.create_virtual_module("meso", "pipeline_meso")
    
    if len(reso.ScanInfo & key) > 0 and len(meso.ScanInfo & key) > 0:
        raise PipelineException(f"Found key in both meso and reso: {key}.")
    elif len(reso.ScanInfo & key) > 0:
        return reso
    elif len(meso.ScanInfo & key) > 0:
        return meso
    else:
        raise PipelineException(f"Could not find key in meso or reso: {key}.")
        
def _get_correct_raster(key):
    """
    Small utility function to return function which corrects Raster phase
    in a pipeline agnostic way. Converts scan to np.float32 if it is not
    already a float dtype.
    """
    pipe = _get_pipe(key)
    raster_phase = (pipe.RasterCorrection & key).fetch1("raster_phase")
    fill_fraction = (pipe.ScanInfo & key).fetch1("fill_fraction")
    if abs(raster_phase) < 1e-7:
        correct_raster = lambda scan: scan.astype(np.float32, copy=False)
    else:
        correct_raster = lambda scan: correct_raster(
            scan, raster_phase, fill_fraction
        )
    return correct_raster
    

def low_memory_motion_correction(scan, raster_phase, fill_fraction, x_shifts, y_shifts):
    """
    Runs an in memory version of our current motion correction found in
    pipeline.utils.galvo_correction. This uses far less memory than the
    parallel motion correction used in motion_correction_method=1.
    """

    chunk_size_in_GB = 1
    single_frame_size = scan[:, :, 0].nbytes
    chunk_size = int(chunk_size_in_GB * 1024**3 / (single_frame_size))

    start_indices = np.arange(0, scan.shape[-1], chunk_size)
    if start_indices[-1] != scan.shape[-1]:
        start_indices = np.insert(start_indices, len(start_indices), scan.shape[-1])

    for start_idx, end_idx in tqdm(
        zip(start_indices[:-1], start_indices[1:]), total=len(start_indices) - 1
    ):

        scan_fragment = scan[:, :, start_idx:end_idx]
        if abs(raster_phase) > 1e-7:
            scan_fragment = correct_raster(
                scan_fragment, raster_phase, fill_fraction
            )  # raster
        scan_fragment = correct_motion(
            scan_fragment, x_shifts[start_idx:end_idx], y_shifts[start_idx:end_idx]
        )  # motion
        scan[:, :, start_idx:end_idx] = scan_fragment

    return scan


def create_template(scan, key):
    """
    Creates the template all frames are compared against to determine the
    amount of motion that occured. Exclusively used for the first iteration
    of motion correction.
    """

    ## Get needed info
    pipe = _get_pipe(key)
    px_height, px_width, _, _ = _get_field_size_info(key)
    skip_rows = int(
        round(px_height * 0.10)
    )  # we discard some rows/cols to avoid edge artifacts
    skip_cols = int(round(px_width * 0.10))

    ## Select template source
    # Default behavior: use middle 2000 frames as template source
    if key["motion_correction_method"] in (1,):
        middle_frame = int(np.floor(scan.shape[-1] / 2))
        mini_scan = scan[
            :,
            skip_rows:-skip_rows,
            skip_cols:-skip_cols,
            :,
            max(middle_frame - 1000, 0) : middle_frame + 1000,
        ]
    # Use full scan as template
    elif key["motion_correction_method"] in (
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
    ):
        mini_scan = scan[
            :,
            skip_rows:-skip_rows,
            skip_cols:-skip_cols,
            :,
            :,
        ]
    else:
        raise PipelineException(f"The create_template() function does not currently support motion_correction_method {key['motion_correction_method']}")

    # Correct mini scan
    correct_raster = _get_correct_raster(key)
    mini_scan = correct_raster(mini_scan)

    # Create template
    mini_scan = 2 * np.sqrt(mini_scan - mini_scan.min() + 3 / 8)  # *
    template = np.mean(mini_scan, axis=-1).squeeze()

    # Apply spatial filtering (if needed)
    if key["motion_correction_method"] in (1,):
        template = ndimage.gaussian_filter(template, 0.7)  # **
    elif key["motion_correction_method"] in (2,3,4,5,6,7,8,9,10,11,12,):
        pass
    else:
        raise PipelineException(f"The create_template() function does not currently support motion_correction_method {key['motion_correction_method']}")
    del mini_scan
    # * Anscombe tranform to normalize noise, increase contrast and decrease outliers' leverage
    # ** Small amount of gaussian smoothing to get rid of high frequency noise

    return template


def create_refined_template(
    scan, x_shifts, y_shifts, key, percentile_thresh=25, smoothing=False
):
    """
    Creates the template all frames are compared against to determine the
    amount of motion that occured. Exclusively used for the second iteration
    of motion correction. Previous x_shifts and y_shifts are expected as
    they are used to filter out frames with too much noise from those used
    to create the template.
    """

    ## Get needed info
    pipe = _get_pipe(key)
    px_height, px_width, _, _ = _get_field_size_info(key)
    skip_rows = int(
        round(px_height * 0.10)
    )  # we discard some rows/cols to avoid edge artifacts
    skip_cols = int(round(px_width * 0.10))

    # Find good frames based on previous motion
    # NOTE: Use <= for threshold since sometimes 0 can be 25th percentile.
    #       < leads to zero good frames in this condition.
    total_shifts = np.sqrt(x_shifts**2 + y_shifts**2)
    good_frame_threshold = np.percentile(total_shifts, percentile_thresh)
    good_frames = total_shifts <= good_frame_threshold

    # Select template source
    mini_scan = scan[:, skip_rows:-skip_rows, skip_cols:-skip_cols, :, good_frames]
    mini_scan = mini_scan.astype(np.float32, copy=False)

    # Correct mini scan
    correct_raster = _get_correct_raster(key)
    mini_scan = correct_raster(mini_scan)

    # Create template
    mini_scan = 2 * np.sqrt(mini_scan - mini_scan.min() + 3 / 8)  # *
    template = np.mean(mini_scan, axis=-1).squeeze()
    if smoothing:
        template = ndimage.gaussian_filter(template, 0.7)  # **
    del mini_scan
    # * Anscombe tranform to normalize noise, increase contrast and decrease outliers' leverage
    # ** Small amount of gaussian smoothing to get rid of high frequency noise

    return template


def process_xy_motion(x_shifts, y_shifts, key):
    """
    Filters raw x_shifts and y_shifts, removing outliers and replacing them
    with interpolated values based on max_y_shift/max_x_shift. Has the
    option of running a momentum based correction, enforcing a max distance
    on amount of movement from one frame to the other.

    NOTE: Momentum correction assumes starting position is median of entire
            motion correction trace to avoid first frame being an outlier.
    """

    # Detect outliers
    pipe = _get_pipe(key)
    _, _, um_per_px_height, um_per_pixel_width = _get_field_size_info(key)
    if key["motion_correction_method"] in (1,):
        max_y_shift, max_x_shift = 20 / np.array([um_per_px_height, um_per_pixel_width])
    elif key["motion_correction_method"] in (2,3,4,5,6,7,8,9,10,11,12,):
        max_y_shift, max_x_shift = 10 / np.array([um_per_px_height, um_per_pixel_width])
    else:
        raise PipelineException(f"The process_xy_motion function does not currently support motion_correction_method {key['motion_correction_method']}")
    y_shifts, x_shifts, outliers = fix_outliers(
        y_shifts, x_shifts, max_y_shift, max_x_shift
    )

    # Center shifts around zero
    y_shifts -= np.median(y_shifts)
    x_shifts -= np.median(x_shifts)

    # Run momentum based correction
    if key["motion_correction_method"] in (1,2,):
        pass
    elif key["motion_correction_method"] in (3,4,5,6,7,8,9,10,11,12,):

        fps = (pipe.ScanInfo & key).fetch1("fps")
        movement_x_max, movement_y_max = 0.5 / np.array([um_per_px_height, um_per_pixel_width]) * 30 / fps

        current_x_coord = np.median(x_shifts)
        x_coords = [current_x_coord]
        for shift in x_shifts[1:]:
            expected_movement = current_x_coord - shift
            if expected_movement < -movement_x_max:
                current_x_coord = current_x_coord + movement_x_max
            elif expected_movement > movement_x_max:
                current_x_coord = current_x_coord - movement_x_max
            else:
                current_x_coord = shift
            x_coords.append(current_x_coord)

        current_y_coord = np.median(y_shifts)
        y_coords = [current_y_coord]
        for shift in y_shifts[1:]:
            expected_movement = current_y_coord - shift
            if expected_movement < -movement_y_max:
                current_y_coord = current_y_coord + movement_y_max
            elif expected_movement > movement_y_max:
                current_y_coord = current_y_coord - movement_y_max
            else:
                current_y_coord = shift
            y_coords.append(current_y_coord)

        x_shifts = np.array(x_coords)
        y_shifts = np.array(y_coords)

    else:
        raise PipelineException(f"The process_xy_motion() function does not currently support motion_correction_method {key['motion_correction_method']}")

    return x_shifts, y_shifts, outliers