from pipeline.meso import CorrectionChannel, RasterCorrection, ScanInfo
from pipeline.utils import performance,galvo_corrections
import scanreader
import numpy as np
from pipeline import experiment

key = {'animal_id':17797,'session':4,'scan_idx':7,'field':1}

"""Computes the motion shifts per frame needed to correct the scan."""
from scipy import ndimage

# Read the scan
scan_filename = (experiment.Scan() & key).local_filenames_as_wildcard
scan = scanreader.read_scan(scan_filename)

# Get some params
px_height, px_width = (ScanInfo.Field() & key).fetch1("px_height", "px_width")
channel = (CorrectionChannel() & key).fetch1("channel") - 1
field_id = key["field"] - 1

# Load some frames from middle of scan to compute template
skip_rows = int(
    round(px_height * 0.10)
)  # we discard some rows/cols to avoid edge artifacts
skip_cols = int(round(px_width * 0.10))
middle_frame = int(np.floor(scan.num_frames / 2))
mini_scan = scan[
    field_id,
    skip_rows:-skip_rows,
    skip_cols:-skip_cols,
    channel,
    max(middle_frame - 1000, 0) : middle_frame + 1000,
]
mini_scan = mini_scan.astype(np.float32, copy=False)

# Correct mini scan
correct_raster = (RasterCorrection() & key).get_correct_raster()
mini_scan = correct_raster(mini_scan)

# Create template
mini_scan = 2 * np.sqrt(mini_scan - mini_scan.min() + 3 / 8)  # *
template = np.mean(mini_scan, axis=-1)
template = ndimage.gaussian_filter(template, 0.7)  # **
# * Anscombe tranform to normalize noise, increase contrast and decrease outliers' leverage
# ** Small amount of gaussian smoothing to get rid of high frequency noise

# Map: compute motion shifts in parallel
f = performance.parallel_motion_shifts  # function to map
raster_phase = (RasterCorrection() & key).fetch1("raster_phase")
fill_fraction = (ScanInfo() & key).fetch1("fill_fraction")
kwargs = {
    "raster_phase": raster_phase,
    "fill_fraction": fill_fraction,
    "template": template,
}
results = performance.map_frames(
    f,
    scan,
    field_id=field_id,
    y=slice(skip_rows, -skip_rows),
    x=slice(skip_cols, -skip_cols),
    channel=channel,
    kwargs=kwargs,
)

# Reduce
y_shifts = np.zeros(scan.num_frames)
x_shifts = np.zeros(scan.num_frames)
for frames, chunk_y_shifts, chunk_x_shifts in results:
    y_shifts[frames] = chunk_y_shifts
    x_shifts[frames] = chunk_x_shifts

# Detect outliers
max_y_shift, max_x_shift = 20 / (ScanInfo.Field() & key).microns_per_pixel
y_shifts, x_shifts, outliers = galvo_corrections.fix_outliers(
    y_shifts, x_shifts, max_y_shift, max_x_shift
)

# Center shifts around zero
y_shifts -= np.median(y_shifts)
x_shifts -= np.median(x_shifts)

# Create results tuple
tuple_ = key.copy()
tuple_["motion_template"] = template
tuple_["y_shifts"] = y_shifts
tuple_["x_shifts"] = x_shifts
tuple_["outlier_frames"] = outliers
tuple_["y_std"] = np.std(y_shifts)
tuple_["x_std"] = np.std(x_shifts)