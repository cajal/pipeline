"""Interface to the CaImAn package (https://github.com/simonsfoundation/CaImAn)"""
import numpy as np
import caiman
import glob, os
import matplotlib.pyplot as plt


def demix_and_deconvolve_with_cnmf(scan, num_components=200, merge_threshold=0.8,
                                   AR_order=2, num_processes=20, block_size=10000,
                                   num_pixels_per_process=10000, init_method='greedy_roi',
                                   neuron_size_in_pixels=10, snmf_alpha=None,
                                   num_background_components=4, init_on_patches=False,
                                   patch_downsampling_factor=None,
                                   percentage_of_patch_overlap=None):
    """ Extract spike train activity directly from the scan using CNMF.

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
    :param int neuron_size_in_pixels: Estimated size of a neuron in the scan (used for
        'greedy_roi' initialization to define the size of the gaussian window)
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
    # Set standard deviation of the gaussian window used in greedy ROI search
    if init_method == 'greedy_roi':
        gaussian_std_dev = [neuron_size_in_pixels // 2, neuron_size_in_pixels // 2]
    else:
        gaussian_std_dev = None # unused

    # Make scan nonnegative
    min_value_in_scan = np.min(scan)
    scan = scan + abs(min_value_in_scan) if min_value_in_scan < 0 else scan

    # Save scan to files (needed to create the memory mapped views below)
    num_timesteps = scan.shape[-1]
    save_size = 10000
    filenames = []
    for i in range(0, num_timesteps, save_size):
        filename = '/tmp/corrected_scan_{}.npy'.format(i)
        chunk = scan[:, :, i: min(i + save_size, num_timesteps)]
        np.save(filename, chunk.transpose([2, 0, 1]))  # save in t x h x w format
        filenames.append(filename)

    # Start the ipyparallel cluster
    client, direct_view, num_processes = caiman.cluster.setup_cluster(
        n_processes=num_processes)

    # Create the small memory mapped files and join them
    mmap_names = caiman.save_memmap_each(filenames, base_name='/tmp/caiman', dview=direct_view)
    mmap_filename = caiman.save_memmap_join(sorted(mmap_names), base_name='/tmp/caiman',
                                            dview=direct_view)

    # 'Load' data
    mmap_scan, scan_dims, num_timesteps = caiman.load_memmap(mmap_filename)
    image_height, image_width = scan_dims
    images = np.reshape(mmap_scan.T, (num_timesteps, *scan_dims), order='F')

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
        cnmf = caiman.cnmf.CNMF(num_processes, only_init_patch=True, p=0,
                                rf=int(round(patch_size / 2)), stride=overlap_in_pixels,
                                k=num_components_per_patch, merge_thresh=merge_threshold,
                                method_init=init_method, gSig=gaussian_std_dev,
                                alpha_snmf=snmf_alpha, gnb=num_background_components,
                                n_pixels_per_process=num_pixels_per_process,
                                block_size=block_size, check_nan=False, dview=direct_view)
        cnmf = cnmf.fit(images)

        # Get results
        initial_A = cnmf.A
        initial_C = cnmf.C
        initial_f = cnmf.f

    # Run CNMF
    cnmf = caiman.cnmf.CNMF(num_processes, k=num_components, method_init=init_method,
                            gSig=gaussian_std_dev, alpha_snmf=snmf_alpha, p=AR_order,
                            merge_thresh=merge_threshold, gnb=num_background_components,
                            check_nan=False, n_pixels_per_process=num_pixels_per_process,
                            block_size=block_size, dview=direct_view, Ain=initial_A,
                            Cin=initial_C, f_in=initial_f)
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

    # Order components by quality (densely distributed in space and high firing)
    location_matrix, activity_matrix = _order_components(location_matrix, activity_matrix)

    # Stop ipyparallel cluster
    client.close()
    caiman.stop_server()

    # Delete log files (one per patch)
    log_files = glob.glob('/tmp/caiman*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)

    # Delete intermediate files (*.mmap and *.npy)
    for filename in filenames + mmap_names + [mmap_filename, '/tmp/caiman.npz']:
        os.remove(filename)

    return (location_matrix, activity_matrix, background_location_matrix,
            background_activity_matrix, raw_traces, spikes, AR_params)


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

    :param np.array location_matrix: (image_height x image_width x timesteps)
    :param np.array background_image: (image_height x image_width).
           Mean or correlation image will look fine.
    """
    # Reshape location_matrix
    image_height, image_width, timesteps = location_matrix.shape
    location_matrix = location_matrix.reshape(-1, timesteps, order='F')

    # Check background matrix was provided, black background otherwise
    if background_image is None:
        background_image = np.zeros([image_height, image_width])

    # Plot contours
    plt.figure()
    caiman.utils.visualization.plot_contours(location_matrix, background_image,
                                             vmin=background_image.min(),
                                             vmax=background_image.max(),
                                             thr_method='nrg', nrgthr=0.99)


def _order_components(location_matrix, activity_matrix):
    """Based on caiman.source_extraction.cnmf.utilities.order_components"""
    # This is the original version from caiman
    # num_components = location_matrix.shape[-1]
    # linear_location = location_matrix.reshape(-1, num_components)
    # density_measure = np.sum(linear_location**4, axis = 0)**(1/4) # small dense masks better
    # norm_density = density_measure / np.linalg.norm(linear_location, axis=0)
    #
    # firing_measure = np.max(activity_matrix.T * np.linalg.norm(linear_location, axis=0),
    #                         axis=0)
    #
    # final_measure = norm_density * firing_measure
    # new_order = np.argsort(final_measure)[::-1]

    # This is good enough (just order them based on the size of the spatial components)
    density_measure = np.linalg.norm(location_matrix, axis=(0, 1))
    new_order = np.argsort(density_measure)[::-1]

    return location_matrix[:, :, new_order], activity_matrix[new_order, :]