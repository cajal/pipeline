"""Interface to the CaImAn package (https://github.com/simonsfoundation/CaImAn)"""
import numpy as np
import caiman
import glob, os
import matplotlib.pyplot as plt


def demix_and_deconvolve_with_cnmf(scan, num_components=100, merge_threshold=0.8,
                                   num_background_components=4, init_on_patches=False,
                                   num_processes=None, memory_usage_in_GB=30,
                                   num_pixels_per_process=10000, block_size=10000,
                                   init_method='greedy_roi', AR_order=2,
                                   neuron_size_in_pixels=10, alpha_snmf=None,
                                   patch_downsampling_factor=4,
                                   percentage_of_patch_overlap=0.2):
    """ Extract spike train activity directly from the scan using CNMF.

    Uses constrained non-negative matrix factorization to find all neurons/components in
    a timeseries of images (locations) and their fluorescence traces (activity) and
    deconvolves them using an autoregressive model of the calcium impulse response
    function (not used for dendrites). See Pnevmatikakis et al., 2016 for details.

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
    :param int memory_usage_in_GB: How much memory to use (may not be exact).
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
    :param int alpha_snmf: Regularization parameter (alpha) for the sparse NMF (if used).
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
    # Calculate standard deviation of the gaussian window used for greedy ROI search
    if init_method == 'greedy_roi':
        gaussian_std_dev = [neuron_size_in_pixels // 2, neuron_size_in_pixels // 2]
    else:
        gaussian_std_dev=None

    # Deal with negative values in the scan.
    min_value_in_scan = np.min(scan)
    scan = scan + abs(min_value_in_scan) if min_value_in_scan < 0 else scan

    # Save scan to files (needed to create the memory mapped views below)
    timesteps = scan.shape[-1]
    save_size = 10000
    filenames = []
    for i in range(0, timesteps, save_size):
        filename = 'corrected_scan_{}.npy'.format(i)
        chunk = scan[:, :, i: min(i + save_size, timesteps)]
        np.save(filename, chunk.transpose([2, 0, 1]))  # save in t x w x h format
        filenames.append(filename)

    # Start the ipyparallel cluster
    client, direct_view, num_processes = caiman.cluster.setup_cluster(
        n_processes=num_processes)

    # Create the small memory mapped files and join them
    mmap_names = caiman.save_memmap_each(filenames, base_name='caiman', dview=direct_view)
    mmap_filename = caiman.save_memmap_join(sorted(mmap_names), base_name='caiman',
                                            dview=direct_view)

    # 'Load' data
    mmap_scan, scan_dims, timesteps = caiman.load_memmap(mmap_filename)
    image_height, image_width = scan_dims
    images = np.reshape(mmap_scan.T, (timesteps, *scan_dims), order='F')

    # Optionally, run the initialization method in small patches to initialize components
    initial_A = None
    initial_C = None
    initial_f = None
    if init_on_patches:
        # Calculate some params
        patch_size = round(image_height / patch_downsampling_factor)  # only square windows
        components_per_patch = max(1, round(num_components / patch_downsampling_factor**2))
        overlap_in_pixels = round(patch_size * percentage_of_patch_overlap)

        # Run CNMF on patches (only for initialization, no impulse response modelling p=0)
        cnmf = caiman.cnmf.CNMF(num_processes, only_init_patch=True, p=0,
                                k=components_per_patch, gnb=num_background_components,
                                gSig=gaussian_std_dev, method_init=init_method,
                                rf=round(patch_size / 2), stride=overlap_in_pixels,
                                merge_thresh=merge_threshold, check_nan=False,
                                memory_fact=memory_usage_in_GB / 16,
                                n_pixels_per_process=num_pixels_per_process,
                                block_size=block_size, dview=direct_view)
        cnmf = cnmf.fit(images)

        # Get results
        initial_A = cnmf.A
        initial_C = cnmf.C
        initial_f = cnmf.f

    # Run CNMF
    cnmf = caiman.cnmf.CNMF(num_processes, k=num_components, method_init=init_method,
                            gSig=gaussian_std_dev, alpha_snmf=alpha_snmf, p=AR_order,
                            merge_thresh=merge_threshold, gnb=num_background_components,
                            memory_fact=memory_usage_in_GB / 16, block_size=block_size,
                            n_pixels_per_process=num_pixels_per_process, check_nan=False,
                            dview=direct_view, Ain=initial_A, Cin=initial_C,
                            f_in=initial_f)
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
    caiman.stop_server()

    # Delete log files (one per patch)
    log_files = glob.glob('caiman*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)

    # Delete intermediate files (*.mmap and *.npy)
    for filename in filenames + mmap_names + [mmap_filename]:
        os.remove(filename)

    return (location_matrix, activity_matrix, background_location_matrix,
            background_activity_matrix, raw_traces, spikes, AR_params)


def compute_correlation_image(scan):
    """ Compute the correlation image for the given scan.

    "The correlation image for each pixel is computed by averaging the correlation
    coefficients (taken over time) of each pixel with its four immediate neighbors."
    (Pnevmatikakis et al., 2016)

    In reality, each pixel trace is normalized through time, traces for one pixel is
    multiplied with that of its four neighbors, this value is averaged over time and
    then averaged over the number of pair multiplications.

    :param np.array scan: 3-dimensional scan (image_height, image_width, timesteps).

    :returns: Correlation image. 2-dimensional array shaped (image_height x image_width).
    :rtype np.array
    """
    scan_as_movie = caiman.movie(scan)
    correlation_image = scan_as_movie.local_correlations(swap_dim=True)

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
                                             vmax=background_image.max())


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





#TODO: Delete this one, already in ExtractRaw
def save_video(scan, location_matrix, activity_matrix, background_location_matrix,
               background_activity_matrix, fps, filename='cnmf_extraction.mp4',
               start_index=0, seconds=30, dpi=200):
    """ Creates an animation video showing the original scan, denoised version, background
    activity and residual scan.

    :param string filename: Output filename (path + filename)
    :param int start_index: Where in the scan to start the video.
    :param int seconds: How long in seconds should the animation run.
    :param int dpi: Dots per inch, controls the quality of the video.

    :returns Figure. You can call show() on it.
    :rtype: matplotlib.figure.Figure
    """
    # TODO: Make this function a part of ExtractRaw (similar to save_video in Prepare)...
    # all required variables from scan to fps could be accessed there from the db

    # Some variables used below
    image_height, image_width, _ = scan.shape
    num_pixels = image_height * image_width
    num_video_frames = round(fps * seconds)

    # Restrict computations to the necessary video frames
    stop_index = start_index + num_video_frames
    scan = scan[:, :, start_index: stop_index]
    activity_matrix = activity_matrix[:, start_index:stop_index]
    background_activity_matrix = background_activity_matrix[:, start_index: stop_index]

    # Calculate matrices
    denoised = np.dot(location_matrix.reshape(num_pixels, -1), activity_matrix)
    denoised = denoised.reshape(image_height, image_width, -1)
    background = np.dot(background_location_matrix.reshape(num_pixels, -1),
                        background_activity_matrix)
    background = background.reshape(image_height, image_width, -1)
    residual = scan - denoised - background

    # Create animation
    import matplotlib.animation as animation

    ## Set the figure
    fig = plt.figure()

    plt.subplot(2, 2, 1)
    plt.title('Original (Y)')
    im1 = plt.imshow(scan[:, :, 0], vmin=scan.min(),
                     vmax=scan.max())  # just a placeholder
    plt.axis('off')
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.title('Denoised (A*C)')
    im2 = plt.imshow(denoised[:, :, 0], vmin=denoised.min(), vmax=denoised.max())
    plt.axis('off')
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.title('Background (B*F)')
    im3 = plt.imshow(background[:, :, 0], vmin=background.min(), vmax=background.max())
    plt.axis('off')
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.title('Residual (Y - A*C - B*F)')
    im4 = plt.imshow(residual[:, :, 0], vmin=residual.min(), vmax=residual.max())
    plt.axis('off')
    plt.colorbar()

    ## Make the animation
    def update_img(i):
        im1.set_data(scan[:, :, i])
        im2.set_data(denoised[:, :, i])
        im3.set_data(background[:, :, i])
        im4.set_data(residual[:, :, i])

    video = animation.FuncAnimation(fig, update_img, num_video_frames,
                                    interval=1000 / fps)

    # Save animation
    print('Saving video at:', filename)
    print('If this takes too long, stop it and call again with dpi < 200 (default)')
    video.save(filename, dpi=dpi)

    return fig

#TODO: Delete. ALready in ExtractRaw
def plot_impulse_responses(AR_params, num_timepoints=100):
    """ Plots the individual impulse response functions assuming an AR(2) process.

    :param np.array AR_params: Parameters (num_components x 2) for the autoregressive process.
    :param int num_timepoints: The number of points after impulse to usse for plotting.

    :returns Figure. You can call show() on it.
    :rtype: matplotlib.figure.Figure
     """

    fig = plt.figure()
    for g1, g2 in AR_params:  # for each component

        # Build impulse response function
        output = np.zeros(num_timepoints)
        output[0] = 1  # initial spike
        output[1] = g1 * output[0]
        for i in range(2, num_timepoints):
            output[i] = g1 * output[i - 1] + g2 * output[i - 2]

        # Plot
        plt.plot(output)

    return fig