import numpy as np
from . import galvo_corrections
import time


def map_frames(f, scan, field_id, y, x, channel, kwargs={}, chunk_size_in_GB=1,
               num_processes=8, queue_size=8):
    """ Apply function f to chunks of the scan (divided in the temporal axis).

    :param function f: Function that receives two positional arguments:
        chunks: A queue with (frames, scan_chunk) tuples. frames is a slice object,
            scan_chunks is a [height, width, num_frames] object
        results: A list to accumulate new results.
    :param Scan scan: An scan object as returned by scanreader.
    :param int field_id: Which field to use: 0-indexed.
    :param slice y: How to slice the scan in y.
    :param slice x: How to slice the scan in x.
    :param int channel: Which channel to read.
    :param dict kwargs: Dictionary with optional kwargs passed to f.
    :param int chunk_size_in_GB: Desired size of each chunk.
    :param int num_processes: Number of processes to use for mapping.
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


def parallel_quality_metrics(chunks, results):
    """ Compute mean intensity per frame, contrast per frame and mean frame.

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
        results.append((frames, mean_intensity, contrast, mean_frame))


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
        y_shifts, x_shifts = galvo_corrections.compute_motion_shifts(chunk, template,
                                                                     num_threads=1)

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


def parallel_fluorescence(chunks, results, raster_phase, fill_fraction, y_shifts,
                         x_shifts, mask_pixels, mask_weights):
    """ Correct scan and compute fluorescence traces for the given masks.

    :param queue chunks: Queue with inputs to consume.
    :param list results: Where to put results.
    :param float raster_phase: Raster phase used for raster correction.
    :param float fill_fraction: Fill fraction used for raster correction.
    :param np.array y_shifts, x_shifts: Motion shifts to correct scan.
    :param list of np.array mask_pixels: Each array is a list of indices where the mask
        is defined. Indices start at 1 and mask has been flattened using F order (Matlab).
    :param list of np.array mask_weights. Each array is the corresponding weights for the
        indices passed in mask_pixels.

    :returns: (traces x num_frames) array. Traces for each mask in this chunk.
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

        # Prepare some params
        image_height, image_width, num_frames = chunk.shape
        flat_chunk = chunk.reshape(-1, num_frames)
        num_masks = len(mask_pixels)

        # Extract signal per mask
        traces = np.zeros([num_masks, num_frames], dtype=np.float32)
        for i, (mp, mw) in enumerate(zip(mask_pixels, mask_weights)):
            mask_as_vector = np.zeros(image_height * image_width, dtype=np.float32)
            mask_as_vector[np.squeeze(mp - 1).astype(int)] = np.squeeze(mw)
            mask = mask_as_vector.reshape(image_height, image_width, order='F')
            traces[i] = np.average(flat_chunk, weights=mask.ravel(), axis=0)

        # Save results
        results.append((frames, traces))



################################## Stacks ##############################################

def map_fields(f, scan, field_ids, channel, y=slice(None), x=slice(None),
               frames=slice(None), kwargs={}, num_processes=8, queue_size=8):
    """ Apply function f to each field in scan

    :param function f: Function that receives two positional arguments:
        fields: A queue with (field_idx, chunk) tuples. field_idx is an integer, chunk is
            a [height, width, frames] array.
        results: A list to accumulate new results.
    :param Scan scan: An scan object as returned by scanreader.
    :param field_ids: List of fields where f will be applied.
    :param slice y: How to slice the scan in y.
    :param slice x: How to slice the scan in x.
    :param int channel: Which channel to read.
    :param slice frames: Frames to pass to f.
    :param dict kwargs: Dictionary with optional kwargs passed to f.
    :param int num_processes: Number of processes to use for mapping.
    :param int queue_size: Maximum size of the queue used to store chunks.

    :returns list results: List with results per field. Order is not guaranteed.
    """
    import multiprocessing as mp

    # Basic checks
    num_processes = min(num_processes, mp.cpu_count() - 1)
    print('Using', num_processes, 'processes')

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
    for i, field_id in enumerate(field_ids):
        chunks.put((i, scan[field_id, y, x, channel, frames])) # field_idx, field tuples

    # Queue STOP signal
    for i in range(num_processes):
        chunks.put((None, None))

    # Wait for processes to finish
    for p in pool:
        p.join()

    return list(results)


def parallel_quality_stack(chunks, results):
    """ Compute mean intensity per frame, contrast per frame, mean

    :param queue chunks: Queue with inputs to consume.
    :param list results: Where to put results.

    :returns: (field_id, mean intensity per frame, contrast (99 -1 percentile) per frame
               and mean frame (average over time in this field) tuple.
    """
    while True:
        # Read next chunk (process locks until something can be read)
        field_idx, field = chunks.get()
        if field is None:  # stop signal when all chunks have been processed
            return

        print(time.ctime(), 'Processing field:', field_idx)

        # Mean intensity
        mean_intensity = np.mean(field, axis=(0, 1), dtype=float)

        # Contrast
        percentiles = np.percentile(field, q=(1, 99), axis=(0, 1))
        contrast = (percentiles[1] - percentiles[0]).astype(float)

        # Mean frame
        mean_frame = np.mean(field, axis=-1, dtype=float)

        # Save results
        results.append((field_idx, mean_intensity, contrast, mean_frame))


def parallel_motion_stack(chunks, results, raster_phase, fill_fraction, skip_rows,
                          skip_cols, max_y_shift, max_x_shift):
    """ Compute motion correction shifts to field in scan.

    Function to run in each process. Consumes input from chunks and writes results to
    results. Stops when stop signal is received in chunks.

    :param queue chunks: Queue with inputs to consume.
    :param list results: Where to put results.
    :param float raster_phase: Raster phase used for raster correction.
    :param float fill_fraction: Fill fraction used for raster correction.
    :param int skip_rows/skip_cols: The number of rows/columss to drop before motion
        corrections. This needs to be an integer greater than zero.
    :param float max_y_shift/max_x_shift: Maximum shifts allowed in outlier detection.

    :returns: (field_id, y_shifts, x_shifts) tuples.
    """
    from scipy import ndimage

    while True:
        # Read next chunk (process locks until something can be read)
        field_idx, field = chunks.get()
        if field is None:  # stop signal when all chunks have been processed
            return

        print(time.ctime(), 'Processing field:', field_idx)

        # Correct raster
        field = field.astype(np.float32, copy=False)
        if abs(raster_phase) > 1e-7:
            field = galvo_corrections.correct_raster(field, raster_phase, fill_fraction)

        # Apply anscombe transform
        field = 2 * np.sqrt(field - field.min() + 3 / 8)

        # Compute initial template by averaging 10 frames that correlate highly with middle one
        num_frames = field.shape[-1]
        frames = np.reshape(field[skip_rows:-skip_rows, skip_cols:-skip_cols], (-1, num_frames)) # num_pixels x num_frames
        residuals = frames - frames.mean(axis=0)
        frames_std = frames.std(axis=0)
        covs = np.mean(residuals.T * residuals[:, int(num_frames / 2)], axis=-1)
        corrs = covs / (frames_std * frames_std[int(num_frames / 2)])
        selected = np.argsort(corrs)[-10:]
        template = ndimage.gaussian_filter(np.mean(field[:, :, selected], axis=-1), 0.6)

        # Compute shifts
        for j in range(3):
            # Compute motion correction shifts (in the cropped up field)
            small_field = field[skip_rows: -skip_rows, skip_cols: - skip_cols]
            small_template = template[skip_rows: -skip_rows, skip_cols: - skip_cols]
            y_shifts, x_shifts = galvo_corrections.compute_motion_shifts(small_field,
                                           small_template, num_threads=1, in_place=False)

            # Fix outliers
            y_shifts, x_shifts, _ = galvo_corrections.fix_outliers(y_shifts, x_shifts,
                                                                   max_y_shift, max_x_shift)

            # Center motions around zero
            y_shifts = y_shifts - np.median(y_shifts)
            x_shifts = x_shifts - np.median(x_shifts)

            # Create template from corrected scan (for next iteration)
            xy_shifts = np.stack([x_shifts, y_shifts])
            corrected = galvo_corrections.correct_motion(field, xy_shifts, in_place=False)
            template = ndimage.gaussian_filter(np.mean(corrected, axis=-1), 0.6)

        # Add to results
        results.append((field_idx, y_shifts, x_shifts))


def parallel_correct_stack(chunks, results, raster_phase, fill_fraction, y_shifts,
                           x_shifts, apply_anscombe=False):
    """ Apply corrections in parallel and return mean of corrected field over time.

    :param queue chunks: Queue with inputs to consume.
    :param list results: Where to put results.
    :param float raster_phase: Raster phase used for raster correction.
    :param float fill_fraction: Fill fraction used for raster correction.
    :param np.array y_shifts: Array with shifts in y for all fields.
    :param np.array x_shifts: Array with shifts in x for all fields
    :param bool apply_anscombe: Whether to apply anscombe transform to the input.

    :returns: (field_id, corrected_field) tuples.
    """
    while True:
        # Read next chunk (process locks until something can be read)
        field_idx, field = chunks.get()
        if field is None:  # stop signal when all chunks have been processed
            return

        print(time.ctime(), 'Processing field:', field_idx)

        # Correct raster
        field = field.astype(np.float32, copy=False)
        if abs(raster_phase) > 1e-7:
            field = galvo_corrections.correct_raster(field, raster_phase, fill_fraction)

        # Correct motion
        xy_shifts = np.stack([x_shifts[field_idx], y_shifts[field_idx]])
        corrected = galvo_corrections.correct_motion(field, xy_shifts)

        # Apply anscombe transform
        if apply_anscombe:
             corrected = 2 * np.sqrt(corrected - corrected.min() + 3 / 8)

        # Average across time
        averaged = np.mean(corrected, axis=-1) if corrected.ndim > 2 else corrected

        # Add to results
        results.append((field_idx, averaged))