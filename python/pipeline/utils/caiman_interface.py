"""Interface to the CaImAn package (https://github.com/simonsfoundation/CaImAn)"""
import numpy as np
import caiman
from caiman.source_extraction.cnmf import cnmf as cnmf
import glob, os
import matplotlib.pyplot as plt


def demix_and_deconvolve_with_cnmf(scan, num_components=200, merge_threshold=0.8,
                                   AR_order=2, num_processes=20, block_size=5000,
                                   num_pixels_per_process=5000, init_method='greedy_roi',
                                   soma_radius_in_pixels=(5, 5), snmf_alpha=None,
                                   num_background_components=4, init_on_patches=False,
                                   patch_downsampling_factor=None,
                                   percentage_of_patch_overlap=None):
    """ Extract spike train activity from two-photon scans using CNMF.

    Uses constrained non-negative matrix factorization to find all neurons/components in
    a timeseries of images (locations) and their fluorescence traces (activity) and
    deconvolves them using an autoregressive model of the calcium impulse response
    function. See Pnevmatikakis et al., 2016 for details.

    Default values work fine for somatic images.

    :param np.array scan: 3-dimensional scan (image_height, image_width, timesteps).
    :param int num_components: An estimate of neurons/spatial components in the scan FOV.
    :param int merge_threshold: Maximum temporal correlation allowed between activity of
            overlapping components before merging them.
    :param int num_background_components:  Number of background components to use.
    :param bool init_on_patches: If True, run the initialization methods on small patches
            of the scan rather than on the whole image.
    :param int num_processes: How many processes to run in parallel. None for as many
            processes as available cores.
    :param int num_pixels_per_process: How many pixels will a process handle each time.
    :param int block_size: 'number of pixels to process at the same time for dot product.'
    :param string init_method: Initialization method for the components.
        'greedy_roi':Look for a gaussian-shaped patch, apply rank-1 NMF, store components,
            calculate residual scan and repeat for num_components.
        'sparse_nmf': Regularized non-negative matrix factorization (as impl. in sklearn)
    :param int AR_order: Order of the autoregressive process used to model the impulse
        response function, e.g., 0 = no modelling; 2 = model rise plus exponential decay.
    :param (float, float) soma_radius_in_pixels: Estimated neuron radius in the scan in
            the y_axis (height) and x_axis (width). Used in'greedy_roi' initialization to 
            define the size of the gaussian window.
    :param int snmf_alpha: Regularization parameter (alpha) for the sparse NMF (if used).
    :param int patch_downsampling_factor: Division to the image dimensions to obtain patch
        dimensions, e.g., if original size is 256 and factor is 10, patches will be 26x26
    :param int percentage_of_patch_overlap: Patches are sampled in a sliding window. This
        controls how much overlap is between adjacent patches (0 for none, 0.9 for 90%)

    :returns Location matrix (image_height x image_width x num_components). Inferred
            location of each component.
    :returns Activity matrix (num_components x timesteps). Inferred fluorescence traces
            (spike train convolved with the fitted impulse response function).
    :returns: Inferred location matrix for background components (image_height x
            image_width x num_background_components).
    :returns: Inferred activity matrix for background components (image_height x
            image_width x num_background_components).
    :returns: Raw fluorescence traces (num_components x timesteps) obtained from the scan
            minus activity from background and other components.
    :returns: Spike matrix (num_components x timesteps). Deconvolved spike activity.
    :returns: Params (num_components x AR_order) for the autoregressive process used to
            model the calcium impulse response of each component:
                   c(t) = c(t-1) * AR_params[0] + c(t-2) * AR_params[1] + ...

    ..note:: Based on code provided by Andrea Giovanucci.
    ..note:: The produced number of components is not exactly what you ask for because
            some components will be merged or deleted.
    ..warning:: Computation- and memory-intensive for big scans.
    """
    # Make scan nonnegative
    min_value_in_scan = np.min(scan)
    scan = scan + abs(min_value_in_scan) if min_value_in_scan < 0 else scan

    # Save as memory mapped file in F order (that's how caiman wants it)
    mmap_filename = save_as_memmap(scan, base_name='/tmp/caiman', order='F')

    # 'Load' scan
    mmap_scan, (image_height, image_width), num_timesteps = caiman.load_memmap(mmap_filename)
    images = np.reshape(mmap_scan.T, (num_timesteps, image_height, image_width), order='F')

    # Start the ipyparallel cluster
    client, direct_view, num_processes = caiman.cluster.setup_cluster(
        n_processes=num_processes)

    # Optionally, run the initialization method in small patches to initialize components
    initial_A = None
    initial_C = None
    initial_f = None
    if init_on_patches:
        # Calculate patch size (only square patches allowed)
        bigger_dimension = max(image_height, image_width)
        smaller_dimension = min(image_height, image_width)
        patch_size = bigger_dimension / patch_downsampling_factor
        patch_size = min(patch_size, smaller_dimension) # smaller than smaller dimension

        # Calculate num_components_per_patch
        num_nonoverlapping_patches = (image_height/patch_size) * (image_width/patch_size)
        num_components_per_patch = num_components / num_nonoverlapping_patches
        num_components_per_patch = max(num_components_per_patch, 1) # at least 1

        # Calculate patch overlap in pixels
        overlap_in_pixels = patch_size * percentage_of_patch_overlap

        # Make sure they are integers
        patch_size = int(round(patch_size))
        num_components_per_patch = int(round(num_components_per_patch))
        overlap_in_pixels = int(round(overlap_in_pixels))

        # Run CNMF on patches (only for initialization, no impulse response modelling p=0)
        cnmf = caiman.source_extraction.cnmf.CNMF(num_processes, only_init_patch=True, p=0,
                                rf=int(round(patch_size / 2)), stride=overlap_in_pixels,
                                k=num_components_per_patch, merge_thresh=merge_threshold,
                                method_init=init_method, gSig=soma_radius_in_pixels,
                                alpha_snmf=snmf_alpha, gnb=num_background_components,
                                n_pixels_per_process=num_pixels_per_process,
                                block_size=block_size, check_nan=False, dview=direct_view,
                                method_deconvolution='cvxpy')
        cnmf = cnmf.fit(images)

        # Delete log files (one per patch)
        log_files = glob.glob('caiman*_LOG_*')
        for log_file in log_files:
            os.remove(log_file)

        # Get results
        initial_A = cnmf.A
        initial_C = cnmf.C
        initial_f = cnmf.f

    # Run CNMF
    cnmf = caiman.source_extraction.cnmf.CNMF(num_processes, k=num_components, method_init=init_method,
                            gSig=soma_radius_in_pixels, alpha_snmf=snmf_alpha, p=AR_order,
                            merge_thresh=merge_threshold, gnb=num_background_components,
                            check_nan=False, n_pixels_per_process=num_pixels_per_process,
                            block_size=block_size, dview=direct_view, Ain=initial_A,
                            Cin=initial_C, f_in=initial_f, method_deconvolution='cvxpy')
    cnmf = cnmf.fit(images)

    # Get final results
    location_matrix = cnmf.A  # pixels x num_components
    activity_matrix = cnmf.C  # num_components x timesteps
    background_location_matrix = cnmf.b  # pixels x num_background_components
    background_activity_matrix = cnmf.f  # num_background_components x timesteps
    spikes = cnmf.S  # num_components x timesteps, spike_ traces
    raw_traces = cnmf.C + cnmf.YrA  # num_components x timesteps
    AR_params = cnmf.g  # AR_order x num_components

    # Reshape spatial matrices to be image_height x image_width x timesteps
    new_shape = (image_height, image_width, -1)
    location_matrix = location_matrix.toarray().reshape(new_shape, order='F')
    background_location_matrix = background_location_matrix.reshape(new_shape, order='F')
    AR_params = np.array(list(AR_params))  # unwrapping it (num_components x 2)

    # Stop ipyparallel cluster
    client.close()
    caiman.stop_server()

    # Delete memory mapped scan
    os.remove(mmap_filename)

    return (location_matrix, activity_matrix, background_location_matrix,
            background_activity_matrix, raw_traces, spikes, AR_params)


def order_components(location_matrix, correlation_image):
    """ Order masks according to brightness and density in the correlation image.

    :param np.array location_matrix: Masks (image_height x image_width x num_components).
    :param array correlation_image: Correlation image (image_height x image_width).

    :return: Indices that would order the location matrix (num_components).
    :rtype: np.array
    """
    # Reshape and normalize masks to sum 1 (num_pixels x num_components)
    reshaped_masks = location_matrix.reshape(-1, location_matrix.shape[-1])
    norm_masks = reshaped_masks / reshaped_masks.sum(axis=0)

    # Calculate correlation_image value weighted by the mask
    quality_measure = (correlation_image.ravel()).dot(norm_masks)

    # Calculate new order according to this measure
    new_order = np.flipud(quality_measure.argsort())  # highest first

    return new_order


def compute_correlation_image(scan):
    """ Compute the correlation image for the given scan.

    Each pixel trace is normalized through time, traces for one pixel are multiplied with
    those of its eight neighbours, these are averaged over time and then averaged over the
    number of pair multiplications (8).

    :param np.array scan: 3-dimensional scan (image_height, image_width, timesteps).

    :returns: Correlation image. 2-dimensional array shaped (image_height x image_width).
    :rtype np.array
    """
    scan_as_movie = caiman.movie(scan)
    correlation_image = scan_as_movie.local_correlations(swap_dim=True,
                                                         eight_neighbours=True)

    return correlation_image


def plot_contours(location_matrix, background_image=None):
    """ Plot each component in location matrix over a background image.

    :param np.array location_matrix: (image_height x image_width x num_components)
    :param np.array background_image: (image_height x image_width). Image for the
        background. Mean or correlation image look fine.
    """
    # Reshape location_matrix
    image_height, image_width, num_components = location_matrix.shape
    location_matrix = location_matrix.reshape(-1, num_components, order='F')

    # Set black background if not provided
    if background_image is None:
        background_image = np.zeros([image_height, image_width])

    # Plot contours
    plt.figure()
    caiman.utils.visualization.plot_contours(location_matrix, background_image,
                                             vmin=background_image.min(),
                                             vmax=background_image.max(),
                                             thr_method='nrg', nrgthr=0.995)


def save_as_memmap(scan, base_name='caiman', order='F'):
    """Save the scan as a memory mapped file as expected by caiman

    :param np.array scan: Scan to save shaped (image_height, image_width, num_timesteps)
    :param string base_name: Base file name for the scan.
    :param string order: Order of the array: either 'C' or 'F'.

    :returns: Filename of the mmap file.
    :rtype: string

    """
    # Get some params
    image_height, image_width, num_timesteps = scan.shape
    num_pixels = image_height * image_width

    # Build filename
    filename = '{}_d1_{}_d2_{}_d3_1_order_{}_frames_{}_.mmap'.format(base_name, image_height,
                                                               image_width, order, num_timesteps)

    # Create memory mapped file
    mmap_file = np.memmap(filename, mode='w+', dtype=np.float32, order=order,
                          shape=(num_pixels, num_timesteps))
    mmap_file[:] = scan.reshape(num_pixels, num_timesteps, order=order)
    mmap_file.flush()

    return filename