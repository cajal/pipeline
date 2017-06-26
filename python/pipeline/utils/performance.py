# TODO: Write and test these

def map_frames(f, scan, field_id, y=slice(None), x=slice(None), channel=0,
               chunk_size_in_GB=10):
    """ Apply function f to chunks of the scan (divided in the temporal axis).

    :param function f: Function that receives a 3-d scan (image_height, image_width, num_frames).
    :param Scan scan: An scan object as returned by scanreader.
    :param int field_id: Which field to use: 0-indexed.
    :param slice y: How to slice the scan in y.
    :param slice x: How to slice the scan in x.
    :param int channel: Which channel to read.
    :param int chunk_size_in_GB: Desired size of each chunk.
    """
    # Get some dimensions
    if scan.is_multiroi:
        image_height = scan.field_heights[field_id]
        image_width = scan.field_widths[field_id]
    else:
        _, image_height, image_width, _, _ = scan.shape
    num_frames = scan.num_frames

    # Calculate the number of frames per chunk
    bytes_per_frame = image_height * image_width * 4 # 4 bytes per pixel
    chunk_size = int(round((chunk_size_in_GB * 2**30) / bytes_per_frame))

    # Apply function over chunks
    for initial_frame in range(0, num_frames, chunk_size):
        frame_slice = slice(initial_frame, initial_frame + chunk_size)

        scan_ = scan[field_id, y, x, channel, frame_slice]
        yield frame_slice, f(scan_)


def map_pixels(f, scan, field_id, channel=0, frames=slice(None), chunk_size_in_GB=10):
    """ Apply function f to chunks of the scan (divided in the y, x axis).

    :param function f: Function that receives a 3-d scan (image_height, image_width, num_frames).
    :param Scan scan: An scan object as returned by scanreader.
    :param int field_id: Which field to use: 0-indexed.
    :param int channel: Which channel to read.
    :param slice frames: How to slice the frames of the scan.
    :param int chunk_size_in_GB: Desired size of each chunk.
    """
    # Get some dimensions
    if scan.is_multiroi:
        image_height = scan.field_heights[field_id]
        image_width = scan.field_widths[field_id]
    else:
        _, image_height, image_width, _, _ = scan.shape
    num_frames = scan.num_frames

    # Calculate the number of frames per chunk
    bytes_per_pixel = num_frames * 4 # 4 bytes per pixel
    chunk_size_y = int(round((chunk_size_in_GB * 2**30) / bytes_per_pixel))
    chunk_size_x= int(round((chunk_size_in_GB * 2**30) / bytes_per_pixel))

    # Apply function over chunks
    for initial_y in range(0, image_height, chunk_size_y):
        for initial_x in range(0, image_width, chunk_size_x):
            yslice = slice(initial_y, initial_y + chunk_size_y)
            xslice = slice(initial_x, initial_x + chunk_size_x)

            scan_ = scan[field_id, yslice, xslice, channel, frames]
            yield yslice, xslice, f(scan_)
