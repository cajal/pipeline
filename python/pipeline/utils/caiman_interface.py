"""Interface to the CaImAn package (https://github.com/simonsfoundation/CaImAn)"""
import numpy as np
import caiman
from caiman.source_extraction.cnmf import cnmf
from caiman.utils import visualization
import glob, os


def demix_and_deconvolve_with_cnmf(scan, num_components=200, AR_order=2,
                                   merge_threshold=0.8, num_processes=20,
                                   num_pixels_per_process=5000, block_size=5000,
                                   num_background_components=4, init_method='greedy_roi',
                                   soma_radius=(5, 5), snmf_alpha=None,
                                   init_on_patches=False, patch_downsampling_factor=None,
                                   percentage_of_patch_overlap=None):
    """ Extract spike train activity from multi-photon scans using CNMF.

    Uses constrained non-negative matrix factorization to find neurons/components
    (locations) and their fluorescence traces (activity) in a timeseries of images, and
    deconvolves them using an autoregressive model of the calcium impulse response
    function. See Pnevmatikakis et al., 2016 for details.

    Default values work alright for somatic images.

    :param np.array scan: 3-dimensional scan (image_height, image_width, num_frames).
    :param int num_components: An estimate of neurons/spatial components in the scan.
    :param int AR_order: Order of the autoregressive process used to model the impulse
        response function, e.g., 0 = no modelling; 2 = model rise plus exponential decay.
    :param int merge_threshold: Maximal temporal correlation allowed between activity of
            overlapping components before merging them.
    :param int num_processes: Number of processes to run in parallel. None for as many
            processes as available cores.
    :param int num_pixels_per_process: Number of pixels that a process handles each
            iteration.
    :param int block_size: 'number of pixels to process at the same time for dot product.'
    :param int num_background_components:  Number of background components to use.
    :param string init_method: Initialization method for the components.
        'greedy_roi':Look for a gaussian-shaped patch, apply rank-1 NMF, store components,
            calculate residual scan and repeat for num_components.
        'sparse_nmf': Regularized non-negative matrix factorization (as impl. in sklearn)
        'local_nmf': ...
    :param (float, float) soma_radius: Estimated neuron radius (in pixels) in y and x.
            Used in'greedy_roi' initialization to define the size of the gaussian window.
    :param int snmf_alpha: Regularization parameter (alpha) for the sparse NMF (if used).
    :param bool init_on_patches: If True, run the initialization methods on small patches
            of the scan rather than on the whole image.
    :param int patch_downsampling_factor: Division to the image dimensions to obtain patch
        dimensions, e.g., if original size is 256 and factor is 10, patches will be 26x26
    :param int percentage_of_patch_overlap: Patches are sampled in a sliding window. This
        controls how much overlap is between adjacent patches (0 for none, 0.9 for 90%)

    :returns Location matrix (image_height x image_width x num_components). Inferred
            location of each component.
    :returns Activity matrix (num_components x num_frames). Inferred fluorescence traces
            (spike train convolved with the fitted impulse response function).
    :returns: Inferred location matrix for background components (image_height x
            image_width x num_background_components).
    :returns: Inferred activity matrix for background components (image_height x
            image_width x num_background_components).
    :returns: Raw fluorescence traces (num_components x num_frames) obtained from the scan
            minus activity from background and other components.
    :returns: Spike matrix (num_components x num_frames). Deconvolved spike activity.
    :returns: Autoregressive process coefficients (num_components x AR_order) used to
            model the calcium impulse response of each component:
                   c(t) = c(t-1) * AR_coeffs[0] + c(t-2) * AR_coeffs[1] + ...

    ..note:: Based on code provided by Andrea Giovanucci.
    ..note:: The produced number of components is not exactly what you ask for because
            some components will be merged or deleted.
    ..warning:: Computation- and memory-intensive for big scans.
    """
    # Save as memory mapped file in F order (that's how caiman wants it)
    mmap_filename = _save_as_memmap(scan, base_name='/tmp/caiman', order='F')

    # 'Load' scan
    mmap_scan, (image_height, image_width), num_frames = caiman.load_memmap(mmap_filename)
    images = np.reshape(mmap_scan.T, (num_frames, image_height, image_width), order='F')

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
        patch_size = min(patch_size, smaller_dimension) # if bigger than small dimension

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
        model = cnmf.CNMF(num_processes, only_init_patch=True, p=0,
                          rf=int(round(patch_size / 2)), stride=overlap_in_pixels,
                          k=num_components_per_patch, merge_thresh=merge_threshold,
                          method_init=init_method, gSig=soma_radius,
                          alpha_snmf=snmf_alpha, gnb=num_background_components,
                          n_pixels_per_process=num_pixels_per_process,
                          block_size=block_size, check_nan=False, dview=direct_view,
                          method_deconvolution='cvxpy')
        model = model.fit(images)

        # Delete log files (one per patch)
        log_files = glob.glob('caiman*_LOG_*')
        for log_file in log_files:
            os.remove(log_file)

        # Get results
        initial_A = model.A
        initial_C = model.C
        initial_f = model.f

    # Run CNMF
    model = cnmf.CNMF(num_processes, k=num_components, p=AR_order,
                      merge_thresh=merge_threshold, gnb=num_background_components,
                      method_init=init_method, gSig=soma_radius, alpha_snmf=snmf_alpha,
                      n_pixels_per_process=num_pixels_per_process, block_size=block_size,
                      check_nan=False, dview=direct_view, Ain=initial_A, Cin=initial_C,
                      f_in=initial_f, method_deconvolution='cvxpy')
    model = model.fit(images)

    # Get final results
    location_matrix = model.A  # pixels x num_components
    activity_matrix = model.C  # num_components x num_frames
    background_location_matrix = model.b  # pixels x num_background_components
    background_activity_matrix = model.f  # num_background_components x num_frames
    spikes = model.S  # num_components x num_frames, spike_ traces
    raw_traces = model.C + model.YrA  # num_components x num_frames
    AR_coefficients = model.g  # AR_order x num_components

    # Reshape spatial matrices to be image_height x image_width x num_frames
    new_shape = (image_height, image_width, -1)
    location_matrix = location_matrix.toarray().reshape(new_shape, order='F')
    background_location_matrix = background_location_matrix.reshape(new_shape, order='F')
    AR_coefficients = np.array(list(AR_coefficients))  # unwrapping it (num_components x 2)

    # Stop ipyparallel cluster
    client.close()
    caiman.stop_server()

    # Delete memory mapped scan
    os.remove(mmap_filename)

    return (location_matrix, activity_matrix, background_location_matrix,
            background_activity_matrix, raw_traces, spikes, AR_coefficients)


def _save_as_memmap(scan, base_name='caiman', order='F'):
    """Save the scan as a memory mapped file as expected by caiman

    :param np.array scan: Scan to save shaped (image_height, image_width, num_frames)
    :param string base_name: Base file name for the scan.
    :param string order: Order of the array: either 'C' or 'F'.

    :returns: Filename of the mmap file.
    :rtype: string

    """
    # Get some params
    image_height, image_width, num_frames = scan.shape
    num_pixels = image_height * image_width

    # Build filename
    filename = '{}_d1_{}_d2_{}_d3_1_order_{}_frames_{}_.mmap'.format(base_name, image_height,
                                                               image_width, order, num_frames)

    # Create memory mapped file
    mmap_file = np.memmap(filename, mode='w+', dtype=np.float32, order=order,
                          shape=(num_pixels, num_frames))
    mmap_file[:] = scan.reshape(num_pixels, num_frames, order=order)
    mmap_file.flush()

    return filename


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


def plot_contours(location_matrix, background_image=None):
    """ Plot each component in location matrix over a background image.

    :param np.array location_matrix: (image_height x image_width x num_components)
    :param np.array background_image: (image_height x image_width). Image for the
        background. Mean or correlation image look fine.
    """
    # Reshape location_matrix
    image_height, image_width, num_components = location_matrix.shape
    location_matrix = location_matrix.reshape(-1, num_components, order='F')

    # Plot contours
    visualization.plot_contours(location_matrix, background_image,
                                vmin=background_image.min(), vmax=background_image.max(),
                                thr_method='nrg', nrgthr=0.995)


def get_centroids(location_matrix):
    """ Use caiman's utility to calculate the centroids of each mask in location matrix.

    :param np.array location_matrix: (image_height x image_width x num_components)

    :returns: Centroids (num_components x 2) in y, x pixels of each component.
    """
    # Reshape location_matrix
    image_height, image_width, num_components = location_matrix.shape
    location_matrix = location_matrix.reshape(-1, num_components, order='F')

    # Get centroids
    fake_background = np.empty([image_height, image_width]) # needed for plot contours
    coordinates = visualization.plot_contours(location_matrix, fake_background)
    import matplotlib.pyplot as plt; plt.close()
    centroids = np.array([coordinate['CoM'] for coordinate in coordinates])

    return centroids