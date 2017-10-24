import numpy as np
from . import galvo_corrections
import time


def map_frames(f, scan, field_id, y, x, channel, kwargs={}, chunk_size_in_GB=1,
               num_processes=8, queue_size=8):
    """ Apply function f to chunks of the scan (divided in the temporal axis).

    :param function f: Function that receives two positional arguments:
        chunks: A queue with (slices, scan_chunk) tuples.
        results: A list to accumulate new results.
    :param Scan scan: An scan object as returned by scanreader.
    :param int field_id: Which field to use: 0-indexed.
    :param slice y: How to slice the scan in y.
    :param slice x: How to slice the scan in x.
    :param int channel: Which channel to read.
    :param dict kwargs: Dictionary with optional kwargs passed to f.
    :param int num_processes: Number of processes to use for mapping.
    :param int chunk_size_in_GB: Desired size of each chunk.
    :param int queue_size: Maximum size of the queue used to store chunks.

    :returns list results: List with results per chunk of scan. Order is not guaranteed.
    """
    import multiprocessing as mp

    # Basic checks
    if chunk_size_in_GB > 2:
        print('Warning: Processing chunks of data bigger than 2 GB could cause timeout '
              'errors when sending data from the master to the working processes.')
    num_processes = min(num_processes, mp.cpu_count() - 1)
    print('Using', num_processes, 'processes')

    # Calculate the number of frames per chunk
    one_frame = scan[field_id, y, x, channel, 0]
    bytes_per_frame = np.prod(one_frame.shape) * 4 # 4 bytes per pixel
    chunk_size = int(round((chunk_size_in_GB * 1024**3) / bytes_per_frame))

#    # Sequential (returns a generator)
#    for i in range(0, num_frames, chunk_size):
#        yield f(scan[field_id, y, x, channel, i: i + chunk_size])

    # Create a Queue to put in new chunks and a list for results
    manager = mp.Manager()
    chunks = manager.Queue(maxsize=queue_size)
    results = manager.list()

    # Start workers (will lock until data appears in chunks)
    pool = []
    for i in range(num_processes):
        p = mp.Process(target=f, args=(chunks, results), kwargs=kwargs)
        p.start()
        pool.append(p)

    # Produce data
    num_frames = scan.num_frames
    for i in range(0, num_frames, chunk_size):
        frames = slice(i, min(i + chunk_size, num_frames))
        chunks.put((frames, scan[field_id, y, x, channel, frames])) # frames, chunk tuples
        # chunks.put(((field_id, y, x, channel, frames), scan.filenames)) # scan_slices, filenames tuples

    # Queue STOP signal
    for i in range(num_processes):
        chunks.put((None, None))

    # Wait for processes to finish
    for p in pool:
        p.join()

    return list(results)


def parallel_motion_shifts(chunks, results, raster_phase, fill_fraction, template):
    """ Compute motion correction shifts to chunks of scan.

    Function to run in each process. Consumes input from chunks and writes results to
    results. Stops when stop signal is received in chunks.

    :param queue chunks: Queue with inputs to consume.
    :param list results: Where to put results.
    :param float raster_phase: Raster phase used for raster correction.
    :param float fill_fraction: Fill fraction used for raster correction.
    :param np.array template: Template used to compute motion shifts.

    :returns: (frames, y_shifts, x_shifts) tuples.
    """
    while True:
        # Read next chunk (process locks until something can be read)
        frames, chunk = chunks.get()
        if chunk is None:  # stop signal when all chunks have been processed
            return

        print(time.ctime(), 'Processing frames:', frames)

        # Correct raster
        chunk = chunk.astype(np.float32, copy=False)
        if abs(raster_phase) > 1e-7:
            chunk = galvo_corrections.correct_raster(chunk, raster_phase, fill_fraction)

        # Compute shifts
        y_shifts, x_shifts, _, _ = galvo_corrections.compute_motion_shifts(chunk, template,
                                num_threads=1, fix_outliers=False, smooth_shifts=False)

        # Add to results
        results.append((frames, y_shifts, x_shifts))


def parallel_summary_images(chunks, results, raster_phase, fill_fraction, y_shifts,
                            x_shifts):
    """ Compute statistics used to compute correlation image and l-6 norm image.

        :param queue chunks: Queue with inputs to consume.
    :param list results: Where to put results.
    :param float raster_phase: Raster phase used for raster correction.
    :param float fill_fraction: Fill fraction used for raster correction.
    :param np.array y_shifts, x_shifts: Motion shifts to correct scan.

    :returns: Sum per pixel, Sum of squared values per pixel, sum of the product of each
        pixel with its 8 neighbors and sum of values of each pixel to the 6th power.
    """
    while True:
        # Read next chunk (process locks until something can be read)
        frames, chunk = chunks.get()
        if chunk is None:  # stop signal when all chunks have been processed
            return

        print(time.ctime(), 'Processing frames:', frames)

        # Correct raster
        chunk = chunk.astype(np.float32, copy=False)
        if abs(raster_phase) > 1e-7:
            chunk = galvo_corrections.correct_raster(chunk, raster_phase, fill_fraction)

        # Correct motion
        xy_motion = np.stack([x_shifts[frames], y_shifts[frames]])
        chunk = galvo_corrections.correct_motion(chunk, xy_motion)

        # Subtract overall brightness per frame
        chunk -= chunk.mean(axis=(0, 1))

        # Compute sum_x and sum_x^2
        chunk_sum = np.sum(chunk, axis=-1, dtype=float)
        chunk_sqsum = np.sum(chunk**2, axis=-1, dtype=float)

        # Compute sum_xy: Multiply each pixel by its eight neighbors
        chunk_xysum = np.zeros((chunk.shape[0], chunk.shape[1], 8))
        for k in [0, 1, 2, 3]: # amount of 90 degree rotations
            rotated_chunk = np.rot90(chunk, k=k)
            rotated_xysum = np.rot90(chunk_xysum, k=k)

            # Multiply each pixel by one above and by one above to the left
            rotated_xysum[1:, :, k] = np.sum(rotated_chunk[1:] * rotated_chunk[:-1], axis=-1, dtype=float)
            rotated_xysum[1:, 1:, 4 + k] = np.sum(rotated_chunk[1:, 1:] * rotated_chunk[:-1, :-1], axis=-1, dtype=float)

            # Return back to original orientation
            chunk = np.rot90(rotated_chunk, k=4 - k)
            chunk_xysum = np.rot90(rotated_xysum, k=4 - k)

        # Compute l6 norm (before square root)
        chunk -= chunk.min()
        chunk_l6norm = np.sum(chunk**6, axis=-1, dtype=float)

        # Save results
        results.append((chunk_sum, chunk_sqsum, chunk_xysum, chunk_l6norm))


def parallel_save_memmap(chunks, results, raster_phase, fill_fraction, y_shifts,
                         x_shifts, mmap_scan):
    """ Correct scan and save in memory mapped file.

    :param queue chunks: Queue with inputs to consume.
    :param list results: Where to put results.
    :param float raster_phase: Raster phase used for raster correction.
    :param float fill_fraction: Fill fraction used for raster correction.
    :param np.array y_shifts, x_shifts: Motion shifts to correct scan.
    :param np.array mmap_scan: Memory mapped file where to write results.

    :returns: Minimum value in chunk. As a side-effect it saves the memory mapped file.
    """
    while True:
        # Read next chunk (process locks until something can be read)
        frames, chunk = chunks.get()
        if chunk is None:  # stop signal when all chunks have been processed
            return

        print(time.ctime(), 'Processing frames:', frames)

        # Correct raster
        chunk = chunk.astype(np.float32, copy=False)
        if abs(raster_phase) > 1e-7:
            chunk = galvo_corrections.correct_raster(chunk, raster_phase, fill_fraction)

        # Correct motion
        xy_motion = np.stack([x_shifts[frames], y_shifts[frames]])
        chunk = galvo_corrections.correct_motion(chunk, xy_motion)

        # Save in mmap scan
        num_frames = chunk.shape[-1]
        mmap_scan[:, frames] = chunk.reshape((-1, num_frames), order='F')
        mmap_scan.flush()

        # Save minimum value in results
        results.append(chunk.min())


def parallel_quality_metrics(chunks, results):
    """ Compute mean intensity per frame, contrast per frame, mean

    :param queue chunks: Queue with inputs to consume.
    :param list results: Where to put results.

    :returns: Mean intensity per frame, contrast (99 -1 percentile) per frame, and mean
        frame (average over time in this chunk).
    """
    while True:
        # Read next chunk (process locks until something can be read)
        frames, chunk = chunks.get()
        if chunk is None:  # stop signal when all chunks have been processed
            return

        print(time.ctime(), 'Processing frames:', frames)

        # Mean intensity
        mean_intensity = np.mean(chunk, axis=(0, 1), dtype=float)

        # Contrast
        percentiles = np.percentile(chunk, q=(1, 99), axis=(0, 1))
        contrast = (percentiles[1] - percentiles[0]).astype(float)

        # Mean frame
        mean_frame = np.mean(chunk, axis=-1, dtype=float)

        # Save results
        results.append((mean_intensity, contrast, mean_frame))




#def map_pixels(f, scan, field_id, channel=0, frames=slice(None), chunk_size_in_GB=10):
#    """ Apply function f to chunks of the scan (divided in the y, x axis).
#
#    :param function f: Function that receives a 3-d scan (image_height, image_width, num_frames).
#    :param Scan scan: An scan object as returned by scanreader.
#    :param int field_id: Which field to use: 0-indexed.
#    :param int channel: Which channel to read.
#    :param slice frames: How to slice the frames of the scan.
#    :param int chunk_size_in_GB: Desired size of each chunk.
#    """
#
#
#    #TODO: skimage.util.apply_parallel(func
#
#    # Get some dimensions
#    if scan.is_multiroi:
#        image_height = scan.field_heights[field_id]
#        image_width = scan.field_widths[field_id]
#    else:
#        _, image_height, image_width, _, _ = scan.shape
#    num_frames = scan.num_frames
#
#    # Calculate the number of frames per chunk
#    bytes_per_pixel = num_frames * 4 # 4 bytes per pixel
#    chunk_size_y = int(round((chunk_size_in_GB * 2**30) / bytes_per_pixel))
#    chunk_size_x= int(round((chunk_size_in_GB * 2**30) / bytes_per_pixel))
#
#    # Apply function over chunks
#    for initial_y in range(0, image_height, chunk_size_y):
#        for initial_x in range(0, image_width, chunk_size_x):
#            yslice = slice(initial_y, initial_y + chunk_size_y)
#            xslice = slice(initial_x, initial_x + chunk_size_x)
#
#            scan_ = scan[field_id, yslice, xslice, channel, frames]
#            yield yslice, xslice, f(scan_)