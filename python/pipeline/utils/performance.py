# -*- coding: utf-8 -*-

def map_frames(function, scan (object), field_id, y=slice(None), x=slice(None), channel, chunk_size_in_GB=10):
    # Get some dimensions
    image_height, image_width, num_frames =

    # Calculate the number of frames per chunk


    bytes_per_frame = 512*512*4 # 4 bytes per pixel
    chunk_size = (chunk_size_in_GB * 2**30) / bytes_per_frame
    for frames in ...
        scan_ = scan[:, :, :, :, frames]
        # preprocess..
        yield frame_slice, function(scan_loaded)

# Divide computation per pixels
def map_pixels
    bytes_per_pixel.
    return yslice, xslice, function(scan)