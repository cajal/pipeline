"""Interface to the CaImAn package (https://github.com/simonsfoundation/CaImAn)."""
import numpy as np
from caiman import cluster, mmapping
from caiman.utils import visualization
from caiman.source_extraction.cnmf import map_reduce, initialization, pre_processing, \
                                          merging, spatial, temporal, deconvolution
import glob, os, time


def log(*messages):
    """ Simple logging function."""
    formatted_time = "[{}]".format(time.ctime())
    print(formatted_time, *messages, flush=True)


def extract_masks(scan, num_components=200, merge_threshold=0.8,
                  num_background_components=4, num_processes=12,
                  num_pixels_per_process=5000, init_method='greedy_roi',
                  soma_radius=(5, 5), snmf_alpha=None, init_on_patches=False,
                  patch_downsampling_factor=None, proportion_patch_overlap=None):
    """ Extract masks from multi-photon scans using CNMF.

    Uses constrained non-negative matrix factorization to find spatial components (masks)
    and their fluorescence traces in a scan. Default values work well for somatic scans.

    Performed operations are:
        [Initialization on full image | Initialization on patches -> merge components] ->
        spatial update -> temporal update -> merge components -> spatial update ->
        temporal update

    :param np.array scan: 3-dimensional scan (image_height, image_width, num_frames).
    :param int num_components: An estimate of the number of spatial components in the scan
    :param int merge_threshold: Maximal temporal correlation allowed between the activity
        of overlapping components before merging them.
    :param int num_background_components:  Number of background components to use.
    :param int num_processes: Number of processes to run in parallel. None for as many
        processes as available cores.
    :param int num_pixels_per_process: Number of pixels that a process handles each
        iteration.
    :param string init_method: Initialization method for the components.
        'greedy_roi': Look for a gaussian-shaped patch, apply rank-1 NMF, store
            components, calculate residual scan and repeat for num_components.
        'sparse_nmf': Regularized non-negative matrix factorization (as impl. in sklearn)
        'local_nmf': ...
    :param (float, float) soma_radius: Estimated neuron radius in y and x (pixels). Used
        in'greedy_roi' initialization to define the size of the gaussian window.
    :param int snmf_alpha: Regularization parameter (alpha) for the sparse NMF (if used).
    :param bool init_on_patches: If True, run the initialization methods on small patches
        of the scan rather than on the whole image.
    :param float patch_downsampling_factor: Image dimensions are divided by this factor
        to obtain patch dimensions, e.g., if original size is 256x256 and factor is 10,
        patches will be 26x26.
    :param float proportion_patch_overlap: Patches are sampled in a sliding window. This
        controls how much overlap is between adjacent patches (0 for none, 0.9 for 90%)

    :returns: Weighted masks (image_height x image_width x num_components). Inferred
        location of each component.
    :returns: Denoised fluorescence traces (num_components x num_frames).
    :returns: Masks for background components (image_height x image_width x
        num_background_components).
    :returns: Traces for background components (image_height x image_width x
        num_background_components).
    :returns: Raw fluorescence traces (num_components x num_frames). Fluorescence of each
        component in the scan minus activity from other components and background.

    ..warning:: The produced number of components is not exactly what you ask for because
        some components will be merged or deleted.
    ..warning:: Better results if scans are nonnegative.
    """
    log('Starting CNMF...')

    # Save as memory mapped file in F order (that's how caiman wants it)
    mmap_filename = _save_as_memmap(scan, base_name='/tmp/caiman', order='F')

    # 'Load' scan
    mmap_scan, (image_height, image_width), num_frames = mmapping.load_memmap(mmap_filename)
    images = np.reshape(mmap_scan.T, (num_frames, image_height, image_width), order='F')

    # Start the ipyparallel cluster
    client, direct_view, num_processes = cluster.setup_cluster(n_processes=num_processes)

    # ******* transcribed from caiman ************
    Y = np.transpose(images, [1, 2, 0]) # (x, y, t)
    Yr = np.transpose(np.reshape(images, (num_frames, -1), order='F')) # (t, pixels)

    # Initialize components
    log('Initializing components...')
    if init_on_patches:
        # TODO: Redo this (per-patch initialization) in a nicer/more efficient way

        # Calculate some parameters
        patch_size = np.array([image_height, image_width]) / patch_downsampling_factor
        num_components_per_patch = num_components / patch_downsampling_factor**2
        num_components_per_patch = max(num_components_per_patch, 1) # at least 1
        overlap_in_pixels = patch_size * proportion_patch_overlap

        # Make sure they are integers
        half_patch_size = np.int32(np.round(patch_size / 2))
        num_components_per_patch = int(round(num_components_per_patch))
        overlap_in_pixels = np.int32(np.round(overlap_in_pixels))

        # Create options dictionary (needed for run_CNMF_patches)
        options = {'patch_params': {'only_init': True, 'remove_very_bad_comps': False, # remove_very_bads_comps unnecesary (same as default)
                                    'ssub': 'UNUSED.', 'tsub': 'UNUSED',
                                    'skip_refinement': 'UNUSED.'}, # remove_very_bads_comps unnecesary (same as default)
                   'preprocess_params': {'check_nan': False}, # check_nan is unnecessary (same as default value)
                   'spatial_params': {'nb': num_background_components}, # nb is unnecessary, it is pased to the function and in init_params
                   'temporal_params': {'p': 0, 'method': 'UNUSED.', 'block_size': 'UNUSED.'},
                   'init_params': {'K': num_components_per_patch, 'gSig': soma_radius,
                                   'method': init_method, 'alpha_snmf': snmf_alpha,
                                   'nb': num_background_components, 'ssub': 1, 'tsub': 1,
                                   'options_local_NMF': 'UNUSED.', 'normalize_init': True},
                                   # ssub, tsub, options_local_NMF, normalize_init unnecessary (same as default values)
                   'merging' : {'thr': 'UNUSED.'}}

        # Initialize per patch
        res = map_reduce.run_CNMF_patches(mmap_filename, (image_height, image_width, num_frames),
                                          options, rf=half_patch_size, stride=overlap_in_pixels,
                                          gnb=num_background_components, dview=direct_view)
        initial_A, initial_C, YrA, initial_b, initial_f, pixels_noise, _ = res

        # Merge spatially overlapping components
        res = merging.merge_components(Yr, initial_A, initial_b, initial_C, initial_f,
                                       initial_C, pixels_noise, {'p': 0, 'method': 'cvxpy'},
                                       None, dview=direct_view, thr=merge_threshold)
        initial_A, initial_C, num_components, merged_ROIs, S, bl, c1, neurons_noise, g = res

        # Delete log files (one per patch)
        log_files = glob.glob('caiman*_LOG_*')
        for log_file in log_files:
            os.remove(log_file)
    else:
        res = initialization.initialize_components(Y, K=num_components, gSig=soma_radius,
                                                   nb=num_background_components,
                                                   method=init_method, alpha_snmf=snmf_alpha)
        initial_A, initial_C, initial_b, initial_f, _ = res

    # Estimate noise per pixel
    log('Calculating noise per pixel...')
    pixels_noise, _ = pre_processing.get_noise_fft_parallel(Yr, num_pixels_per_process,
                                                            direct_view)

    # Update masks
    log('Updating masks...')
    A, b, C, f = spatial.update_spatial_components(Yr, initial_C, initial_f, initial_A,
                                                   sn=pixels_noise, dims=(image_height, image_width),
                                                   method='dilate', dview=direct_view,
                                                   n_pixels_per_process=num_pixels_per_process,
                                                   nb=num_background_components,
                                                   method_ls='lasso_lars')

    # Update traces (no impulse response modelling p=0)
    log('Updating traces...')
    res = temporal.update_temporal_components(Yr, A, b, C, f, nb=num_background_components,
                                              block_size=10000, p=0, method='cvxpy',
                                              dview=direct_view)
    C, A, b, f, S, bl, c1, neurons_noise, g, YrA = res


    # Merge components
    log('Merging overlapping (and temporally correlated) masks...')
    res = merging.merge_components(Yr, A, b, C, f, S, pixels_noise, {'p': 0, 'method': 'cvxpy'},
                                   None, dview=direct_view, thr=merge_threshold, bl=bl,
                                   c1=c1, sn=neurons_noise, g=g)
    A, C, num_components, merged_pairs, S, bl, c1, neurons_noise, g = res

    # Refine masks
    log('Refining masks...')
    A, b, C, f = spatial.update_spatial_components(Yr, C, f, A, sn=pixels_noise,
                                                   dims=(image_height, image_width),
                                                   method='dilate', dview=direct_view,
                                                   n_pixels_per_process=num_pixels_per_process,
                                                   nb=num_background_components,
                                                   method_ls='lasso_lars')

    # Refine traces
    log('Refining traces...')
    res = temporal.update_temporal_components(Yr, A, b, C, f, nb=num_background_components,
                                              block_size=10000, p=0, method='cvxpy',
                                              dview=direct_view)
    C, A, b, f, S, bl, c1, neurons_noise, g, YrA = res

    log('Done.')
    #*********************

    # Stop ipyparallel cluster
    client.close()
    cluster.stop_server()

    # Delete memory mapped scan
    os.remove(mmap_filename)

    # Get results
    masks = A.toarray().reshape((image_height, image_width, -1), order='F') # h x w x num_components
    traces = C  # num_components x num_frames
    background_masks = b.reshape((image_height, image_width, -1), order='F') # h x w x num_components
    background_traces = f  # num_background_components x num_frames
    raw_traces = C + YrA  # num_components x num_frames

    # Rescale traces to match scan range (~ np.average(trace*mask, weights=mask))
    scaling_factor = np.sum(masks**2, axis=(0, 1)) / np.sum(masks, axis=(0, 1))
    traces = traces * np.expand_dims(scaling_factor, -1)
    raw_traces = raw_traces * np.expand_dims(scaling_factor, -1)
    masks = masks / scaling_factor
    background_scaling_factor = np.sum(background_masks**2, axis=(0, 1)) / np.sum(background_masks,
                                                                                  axis=(0,1))
    background_traces = background_traces * np.expand_dims(background_scaling_factor, -1)
    background_masks = background_masks / background_scaling_factor

    return masks, traces, background_masks, background_traces, raw_traces


def _save_as_memmap(scan, base_name='caiman', order='C'):
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


def deconvolve(trace, AR_order=2):
    """ Deconvolve traces using noise constrained deconvolution (Pnevmatikakis et al., 2016)

    :param np.array trace: 1-d array (num_frames) with the fluorescence trace.
    :param int AR_order: Order of the autoregressive process used to model the impulse
        response function, e.g., 0 = no modelling; 2 = model rise plus exponential decay.

    :returns: Deconvolved spike trace.
    :returns: AR coefficients (AR_order) that model the calcium response:
            c(t) = c(t-1) * AR_coeffs[0] + c(t-2) * AR_coeffs[1] + ...
    """
    _, _, _, AR_coeffs, _, spike_trace = deconvolution.constrained_foopsi(trace, p=AR_order,
        method='cvxpy', bas_nonneg=False, fudge_factor=0.96) # fudge_factor is a regularization term

    return spike_trace, AR_coeffs


def order_components(masks, correlation_image):
    """ Order masks according to brightness and density in the correlation image.

    :param np.array masks: Masks (image_height x image_width x num_components).
    :param array correlation_image: Correlation image (image_height x image_width).

    :return: Indices that would order the masks (num_components).
    :rtype: np.array
    """
    # Reshape and normalize masks to sum 1 (num_pixels x num_components)
    reshaped_masks = masks.reshape(-1, masks.shape[-1])
    norm_masks = reshaped_masks / reshaped_masks.sum(axis=0)

    # Calculate correlation_image value weighted by the mask
    quality_measure = (correlation_image.ravel()).dot(norm_masks)

    # Calculate new order according to this measure
    new_order = np.flipud(quality_measure.argsort())  # highest first

    return new_order


def plot_masks(masks, background_image):
    """ Plot masks over a background image.

    :param np.array masks: Masks (image_height x image_width x num_components)
    :param np.array background_image: Image (image_height x image_width) to plot in the
        background. Mean or correlation image look fine.
    """
    # Reshape masks
    image_height, image_width, num_components = masks.shape
    masks = masks.reshape(-1, num_components, order='F')

    # Plot contours
    visualization.plot_contours(masks, background_image, vmin=background_image.min(),
                                vmax=background_image.max(), thr_method='nrg', nrgthr=0.995)


def get_centroids(masks):
    """ Calculate the centroids of each mask (calls caiman's plot_contours).

    :param np.array masks: Masks (image_height x image_width x num_components)

    :returns: Centroids (num_components x 2) in y, x pixels of each component.
    """
    # Reshape masks
    image_height, image_width, num_components = masks.shape
    masks = masks.reshape(-1, num_components, order='F')

    # Get centroids
    fake_background = np.empty([image_height, image_width]) # needed for plot contours
    coordinates = visualization.plot_contours(masks, fake_background)
    import matplotlib.pyplot as plt; plt.close()
    centroids = np.array([coordinate['CoM'] for coordinate in coordinates])

    return centroids
























# Legacy: Used in preprocess.ExtractRaw
def demix_and_deconvolve_with_cnmf(scan, num_components=200, AR_order=2,
                                   merge_threshold=0.8, num_processes=20,
                                   num_pixels_per_process=5000, block_size=10000,
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
    :param int block_size: 'number of pixels to process at the same time for dot product'
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
    :returns: Raw fluorescence traces (num_components x num_frames) obtained from the
        scan minus activity from background and other components.
    :returns: Spike matrix (num_components x num_frames). Deconvolved spike activity.
    :returns: Autoregressive process coefficients (num_components x AR_order) used to
        model the calcium impulse response of each component:
            c(t) = c(t-1) * AR_coeffs[0] + c(t-2) * AR_coeffs[1] + ...

    ..note:: Based on code provided by Andrea Giovanucci.
    ..note:: The produced number of components is not exactly what you ask for because
        some components will be merged or deleted.
    ..warning:: Computation- and memory-intensive for big scans.
    """
    import caiman
    from caiman.source_extraction.cnmf import cnmf

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