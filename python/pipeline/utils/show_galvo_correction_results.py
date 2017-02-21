# Written by: Erick Cobos
# Date: 20-Feb-2017

""" Little script to load, correct, show and save a scan."""
from .. import preprocess
from .. import experiment
from commons import lab
import os.path
from tiffreader import TIFFReader
import numpy as np
from pipeline.utils import galvo_corrections
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set some params
number_of_samples = 1000 # Number of frames to show in the video
key = {'animal_id':10804, 'session':3, 'scan_idx':1}#1,2,3} # scan to work with (jiakun-approved)

# Get local file path
scan_path = (experiment.Session() & key).fetch1['scan_path']
local_path  = lab.Paths().get_local_path(scan_path)
filename = (experiment.Scan() & key).fetch['filename'][0] # not sure why the 0 here
#local_filename = os.path.join(local_path, filename) + '_*.tif' # got 36 parts
local_filename = os.path.join(local_path, filename) + '_00001.tif' # only one part

# Get raster_correction and motion_correction params
raster_phase, fill_fraction= (preprocess.Prepare.Galvo() & key).fetch1['raster_phase',
                                                                       'fill_fraction']
xy_motion = (preprocess.Prepare.GalvoMotion() & key).fetch1['motion_xy']
fps = (preprocess.Prepare.Galvo() & key).fetch1['fps'] # for video showing

# Load it
reader = TIFFReader(local_filename)
scan = np.double(reader[:, :, :, :, :]) # not generalizable
#original_shape = reader.shape
#scan = np.double(reader).reshape( (original_shape[0], original_shape[1], :) )

# Preserve a sample from the original
original_sample = (scan[:, :, 0, 0, :number_of_samples]).copy()

# Raster correction
raster_corrected = galvo_corrections.correct_raster(scan, raster_phase, fill_fraction)

# Motion correction
motion_corrected = galvo_corrections.correct_motion(raster_corrected, xy_motion)

# Get a corrected sample
corrected_sample = motion_corrected[:, :, 0, 0, :number_of_samples]

# Show a video to check everything is OK
fig = plt.figure()
plt.subplot(1,2,1); plt.title('Original');
im1 = plt.imshow(original_sample[:,:,0]) # just a placeholder
plt.subplot(1,2,2); plt.title('Corrected')
im2 = plt.imshow(original_sample[:,:,0]) # just a placeholder
def update_img(i):
    im1.set_data(original_sample[:, :, i])
    im2.set_data(corrected_sample[:, :, i])
video = animation.FuncAnimation(fig, update_img, number_of_samples, interval=fps)
plt.show()

# Save video
video.save('galvo_corrections.mp4')

## For Andrea
# Make it 3-d, put time in first axis (array ends up being t x w x h)
final_scan = motion_corrected[:,:,0,0,:].swapaxes(0,-1) # has only one channel

assert(final_scan.ndim == 3)

# Save to h5
#import h5py
#with h5py.File('scan.h5', 'w') as scan_file:
#    scan_file.create_dataset('scan', data=final_scan)

# Or save to numpy
np.save('scan.npy', final_scan)
