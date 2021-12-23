#!/usr/local/bin/python3
import logging 
from pipeline.meso import *
from tqdm import tqdm


# PATCH DELETION OF meso.SummaryImages.Correlation and meso.SummaryImages.L6Norm BY MARIO GALDAMEZ ON 2021-12-17


logging.basicConfig(level=logging.ERROR)
logging.getLogger('datajoint.connection').setLevel(logging.DEBUG)
if hasattr(dj.connection, 'query_log_max_length'):
    dj.connection.query_log_max_length = 3000

keys = MotionCorrection & (SummaryImages - SummaryImages.Correlation) & {"pipe_version": 1}

for key in tqdm(keys):
    # Read the scan
    scan_filename = (experiment.Scan() & key).local_filenames_as_wildcard
    scan = scanreader.read_scan(scan_filename)

    for channel in range(scan.num_channels):
        # Map: Compute some statistics in different chunks of the scan
        f = performance.parallel_summary_images  # function to map
        raster_phase = (RasterCorrection() & key).fetch1("raster_phase")
        fill_fraction = (ScanInfo() & key).fetch1("fill_fraction")
        y_shifts, x_shifts = (MotionCorrection() & key).fetch1(
            "y_shifts", "x_shifts"
        )
        kwargs = {
            "raster_phase": raster_phase,
            "fill_fraction": fill_fraction,
            "y_shifts": y_shifts,
            "x_shifts": x_shifts,
        }
        results = performance.map_frames(
            f, scan, field_id=key["field"] - 1, channel=channel, kwargs=kwargs
        )

        # Reduce: Compute average images
        l6norm_image = np.sum([r[1] for r in results], axis=0) ** (1 / 6)

        # Reduce: Compute correlation image
        sum_x = np.sum([r[2] for r in results], axis=0)  # h x w
        sum_sqx = np.sum([r[3] for r in results], axis=0)  # h x w
        sum_xy = np.sum([r[4] for r in results], axis=0)  # h x w x 8
        denom_factor = np.sqrt(scan.num_frames * sum_sqx - sum_x ** 2)
        corrs = np.zeros(sum_xy.shape)
        for k in [0, 1, 2, 3]:
            rotated_corrs = np.rot90(corrs, k=k)
            rotated_sum_x = np.rot90(sum_x, k=k)
            rotated_dfactor = np.rot90(denom_factor, k=k)
            rotated_sum_xy = np.rot90(sum_xy, k=k)

            # Compute correlation
            rotated_corrs[1:, :, k] = (
                scan.num_frames * rotated_sum_xy[1:, :, k]
                - rotated_sum_x[1:] * rotated_sum_x[:-1]
            ) / (rotated_dfactor[1:] * rotated_dfactor[:-1])
            rotated_corrs[1:, 1:, 4 + k] = (
                scan.num_frames * rotated_sum_xy[1:, 1:, 4 + k]
                - rotated_sum_x[1:, 1:] * rotated_sum_x[:-1, :-1]
            ) / (rotated_dfactor[1:, 1:] * rotated_dfactor[:-1, :-1])

            # Return back to original orientation
            corrs = np.rot90(rotated_corrs, k=4 - k)

        correlation_image = np.sum(corrs, axis=-1)
        norm_factor = 5 * np.ones(correlation_image.shape)  # edges
        norm_factor[[0, -1, 0, -1], [0, -1, -1, 0]] = 3  # corners
        norm_factor[1:-1, 1:-1] = 8  # center
        correlation_image /= norm_factor

        # Insert
        field_key = {**key, "channel": channel + 1}
        SummaryImages.L6Norm().insert1(
            {**field_key, "l6norm_image": l6norm_image}
        )
        SummaryImages.Correlation().insert1(
            {**field_key, "correlation_image": correlation_image}
        )