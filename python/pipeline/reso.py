import os
from warnings import warn
import datajoint as dj
import numpy as np
import sh
from commons import lab
import scanreader

from . import experiment, config, PipelineException
from .utils.dsp import mirrconv
from .utils.eye_tracking import ROIGrabber, ts2sec, read_video_hdf5, CVROIGrabber
from .utils import galvo_corrections
import matplotlib.pyplot as plt

from distutils.version import StrictVersion

assert StrictVersion(dj.__version__) >= StrictVersion('0.2.9')

schema = dj.schema('pipeline_meso', locals())


# TODO: find a solution to figure out which channel to use for correction
# TODO: find a solution to figure out which segmentation to use

# TODO: @Erick
# - fill Spikes tables and its part table from Segmentation



def notnan(x, start=0, increment=1):
    while np.isnan(x[start]) and 0 <= start < len(x):
        start += increment
    return start


def fill_nans(x):
    """
    :param x:  1D array  -- will
    :return: the array with nans interpolated
    The input argument is modified.
    """
    nans = np.isnan(x)
    x[nans] = 0 if nans.all() else np.interp(nans.nonzero()[0], (~nans).nonzero()[0], x[~nans])
    return x


def normalize(img):
    return (img - img.min()) / (img.max() - img.min())


def erd():
    """a shortcut for convenience"""
    dj.ERD(schema).draw(prefix=False)


@schema
class Slice(dj.Lookup):
    definition = """  # slices in resonant scanner scans
    slice  : tinyint  # slice in scan
    """
    contents = ((i,) for i in range(12))


@schema
class Channel(dj.Lookup):
    definition = """  # recording channel, directly related to experiment.PMTFilterSet.Channel
    channel : tinyint
    """
    contents = [[1], [2], [3], [4]]


@schema
class SegmentationMethod(dj.Lookup):
    definition = """  
    #  methods for extraction from raw data for either AOD or Galvo data

    extract_method :  tinyint
    ---
    segmentation  :  varchar(16)   #
    """

    @property
    def contents(self):
        yield from zip([1, 2], ['manual', 'nmf'])


@schema
class SpikeMethod(dj.Lookup):
    definition = """
    spike_method   :  smallint   # spike inference method
    ---
    spike_method_name     : varchar(16)   #  short name to identify the spike inference method
    spike_method_details  : varchar(255)  #  more details about
    language :  enum('matlab','python')   #  implementation language
    """

    contents = [
        [2, "oopsi", "nonnegative sparse deconvolution from Vogelstein (2010)", "python"],
        [3, "stm", "spike triggered mixture model from Theis et al. (2016)", "python"],
        [5, "nmf", "", "python"]
    ]

    def spike_traces(self, X, fps):
        try:
            import c2s
        except ImportError:
            warn("c2s was not found. You won't be able to populate ExtracSpikes")
        assert self.fetch1['language'] == 'python', "This tuple cannot be computed in python."
        if self.fetch1['spike_method'] == 3:
            N = len(X)
            for i, trace in enumerate(X):
                print('Predicting trace %i/%i' % (i + 1, N))
                tr0 = np.array(trace.pop('trace').squeeze())
                start = notnan(tr0)
                end = notnan(tr0, len(tr0) - 1, increment=-1)
                trace['calcium'] = np.atleast_2d(tr0[start:end + 1])

                trace['fps'] = fps
                data = c2s.preprocess([trace], fps=fps)
                data = c2s.predict(data, verbosity=0)

                tr0[start:end + 1] = data[0].pop('predictions')
                data[0]['rate_trace'] = tr0.T
                data[0].pop('calcium')
                data[0].pop('fps')

                yield data[0]


@schema
class ScanInfo(dj.Imported):
    definition = """  # master table that gathers data about the scans of different types, prepares for trace extraction
    -> experiment.Scan
    ---
    nframes_requested       : int               # number of volumes (from header)
    nframes                 : int               # frames recorded
    px_width                : smallint          # pixels per line
    px_height               : smallint          # lines per frame
    um_width                : float             # width in microns
    um_height               : float             # height in microns
    bidirectional           : tinyint           # 1=bidirectional scanning
    fps                     : float             # (Hz) frames per second
    zoom                    : decimal(4,1)      # zoom factor
    dwell_time              : float             # (us) microseconds per pixel per frame
    nchannels               : tinyint           # number of recorded channels
    nslices                 : tinyint           # number of slices
    slice_pitch             : float             # (um) distance between slices
    fill_fraction           : float             # raster scan temporal fill fraction (see scanimage)
    """

    @property
    def key_source(self):
        return (experiment.Scan() - experiment.ScanIgnored()).proj()

    def estimate_num_components_per_slice(self):
        """ Estimates the number of components per scan slice using simple rules of thumb.

        For somatic scans, estimate number of neurons based on:
        (100x100x100)um^3 = 1e6 um^3 -> 1e2 neurons; (1x1x1)mm^3 = 1e9 um^3 -> 1e5 neurons

        For axonal/dendritic scans, just ten times our estimate of neurons.

        :returns: Number of components
        :rtype: int
        """

        # Get slice dimensions (in micrometers)
        slice_height, slice_width = self.fetch1['um_height', 'um_width']
        slice_thickness = 10  # assumption
        slice_volume = slice_width * slice_height * slice_thickness

        # Estimate number of components
        if experiment.Session.TargetStructure() & self:  # scan is axonal/dendritic
            num_components = slice_volume * 0.001  # ten times as many neurons
        else:
            num_components = slice_volume * 0.0001

        return int(round(num_components))

    def estimate_soma_radius_in_pixels(self):
        """ Estimates the radius of a neuron in the scan (in pixels). Assumes soma is
         14 x 14 microns.

         :returns: a tuple with the estimated pixel radius on the y-axis (height) and
            x-axis (width) of the scan.
         :rtype: tuple of floats
        """
        soma_radius_in_microns = 7  # assumption

        # Calculate size in pixels (height radius)
        um_height, px_height = self.fetch1['um_height', 'px_height']
        height_microns_per_pixel = um_height / px_height
        height_radius_in_pixels = soma_radius_in_microns / height_microns_per_pixel

        # Calculate size in pixels (width radius)
        um_width, px_width = self.fetch1['um_width', 'px_width']
        width_microns_per_pixel = um_width / px_width
        width_radius_in_pixels = soma_radius_in_microns / width_microns_per_pixel

        return (height_radius_in_pixels, width_radius_in_pixels)

    def _make_tuples(self, key):
        """ Read some scan parameters, compute FOV in microns and raster phase for 
        raster correction. 

        :param scan Scan: The scan. An Scan object returned by scanreader.
        """
        # Read the scan
        scan_filename = (experiment.Scan() & key).local_filenames_as_wildcard
        scan = scanreader.read_scan(scan_filename)

        # Get attributes
        key = key.copy()  # in case key is reused somewhere else
        key['nframes_requested'] = scan.num_requested_frames
        key['nframes'] = scan.num_frames
        key['px_height'] = scan.image_height
        key['px_width'] = scan.image_width
        key['bidirectional'] = scan.is_bidirectional
        key['fps'] = scan.fps
        key['zoom'] = scan.zoom
        key['dwell_time'] = (scan.seconds_per_line / scan._page_width) * 1e6
        key['nchannels'] = scan.num_channels
        key['nslices'] = scan.num_fields
        key['slice_pitch'] = scan.zstep_in_microns
        key['fill_fraction'] = scan.temporal_fill_fraction

        # Calculate height and width in microns
        # Estimate using measured FOVs for similar setups
        fov_rel = (experiment.FOV() * experiment.Session() * experiment.Scan()
                   & key & 'session_date>=fov_ts')
        zooms = fov_rel.fetch['mag'].astype(np.float32)  # measured zooms in setup
        closest_zoom = zooms[np.argmin(np.abs(np.log(zooms / scan.zoom)))]
        um_height, um_width = (fov_rel & {'mag': closest_zoom}).fetch1['height', 'width']
        key['um_height'] = float(um_height) * (closest_zoom / scan.zoom) * scan._y_angle_scale_factor
        key['um_width'] = float(um_width) * (closest_zoom / scan.zoom) * scan._x_angle_scale_factor

        self.insert1(key)


@schema
class RasterCorrection(dj.Computed):
    definition = """
    # computes information for raster correction
    
    ->ScanInfo
    ---
    preview_frame           : longblob          # raw average frame from channel 1 from an early fragment of the movie
    raster_phase            : float             # shift of odd vs even raster lines
    """

    def get_correct_raster(self):
        """
         :returns: A function to perform raster correction on the scan
                [image_height, image_width, channels, slices, num_frames].
        """
        raster_phase, fill_fraction = (ScanInfo() * self).fetch1['raster_phase', 'fill_fraction']
        if raster_phase == 0:
            return lambda scan: np.double(scan)
        else:
            return lambda scan: galvo_corrections.correct_raster(scan, raster_phase,
                                                                 fill_fraction)

    def _make_tuples(self, key):
        # Read the scan
        scan_filename = (experiment.Scan() & key).local_filenames_as_wildcard
        scan = scanreader.read_scan(scan_filename)

        # Select channel to use for raster and motion correction according to dye used
        fluorophore = (experiment.Session.Fluorophore() & key).fetch1['fluorophore']
        channel = 1 if fluorophore in ['RCaMP1a', 'mCherry', 'tdTomato'] else 0
        if fluorophore == 'Twitch2B':
            print('Warning: Twitch2B scan. Using first channel to compute correction '
                  'parameters.')

        # Compute a preview image of the scan: mean of frames 1000-3000
        preview_field = int(np.floor(scan.num_fields / 2))
        if scan.num_frames < 2000:
            mini_field = scan[preview_field, :, :, channel, -2000:]
        else:
            mini_field = scan[preview_field, :, :, channel, 1000:3000]
        preview_image = np.mean(mini_field, axis=-1)
        key['preview_frame'] = preview_image

        # Compute raster correction parameters
        if scan.is_bidirectional and scan.scanner_type == 'Resonant':
            key['raster_phase'] = galvo_corrections.compute_raster_phase(preview_image,
                                                                         scan.temporal_fill_fraction)
        else:
            key['raster_phase'] = 0

        # Insert result
        self.insert1(key)


@schema
class MotionCorrection(dj.Computed):
    definition = """   
    # motion correction for galvo scans
    
    -> RasterCorrection
    -> Slice
    ---
    -> Channel
    template                    : longblob       # stack that was used as alignment template
    motion_xy                   : longblob       # (pixels) x,y motion correction offsets
    motion_rms                  : float          # (um) stddev of motion
    align_times=CURRENT_TIMESTAMP: timestamp     # automatic
    """

    def get_correct_motion(self):
        """
        :returns: A function to performs motion correction on scans
                  [image_height, image_width, channels, slices, num_frames].
        """
        xy_motion = self.fetch1['motion_xy']

        def my_lambda_function(scan, indices=None):
            if indices is None:
                return galvo_corrections.correct_motion(scan, xy_motion)
            else:
                return galvo_corrections.correct_motion(scan, xy_motion[:, indices])

        return my_lambda_function

    def _make_tuples(self, key):
        """Computes the motion shifts per frame needed to correct the scan."""

        # Read the scan
        scan_filename = (experiment.Scan() & key).local_filenames_as_wildcard
        scan = scanreader.read_scan(scan_filename)

        # Select channel to use for raster and motion correction according to dye used
        fluorophore = (experiment.Session.Fluorophore() & key).fetch1['fluorophore']
        channel = 1 if fluorophore in ['RCaMP1a', 'mCherry', 'tdTomato'] else 0
        if fluorophore == 'Twitch2B':
            print('Warning: Twitch2B scan. Using first channel to compute correction '
                  'parameters.')

        key['channel'] = channel + 1  # indices start at 1 in database

        # Get some params
        um_height, px_height, um_width, px_width = \
            (ScanInfo() & key).fetch1['um_height', 'px_height', 'um_width', 'px_width']

        # Get raster correction function
        correct_raster = (RasterCorrection() & key).get_correct_raster()

        for field_id in range(scan.num_fields):
            print('Correcting field', field_id + 1)
            field = scan[field_id, :, :, channel, :]  # 3-d (height, width, frames)
            key['slice'] = field_id + 1

            # Correct raster effects (needed for subpixel changes in y)
            field = correct_raster(field)

            # Create template
            if scan.num_frames < 3000:
                mini_field = field[:, :, -2000:]
            else:
                mini_field = field[:, :, 1000:3000]
            template = np.mean(mini_field, axis=-1)
            template -= template.min()  # set lowest element to zero
            template = 2 * np.sqrt(
                template + 3 / 8)  # anscombe transform: decrease leverage of outliers and increase contrast
            key['template'] = template

            # Get motion correction shifts
            y_shifts, x_shifts = galvo_corrections.compute_motion_shifts(field, template)
            key['motion_xy'] = np.stack([x_shifts, y_shifts])

            # Calculate root mean squared distance of motion shifts
            y_shifts_in_microns = y_shifts * (um_height / px_height)
            x_shifts_in_microns = x_shifts * (um_width / px_width)
            # x_distances = x_shifts_in_microns - x_shifts_in_microns.mean()
            # y_distances = y_shifts_in_microns - y_shifts_in_microns.mean()
            # key['motion_rms'] = np.sqrt(np.mean(np.square([x_distances, y_distances])))

            # Calculate mean euclidean distance
            key['motion_rms'] = np.mean(np.sqrt(x_shifts_in_microns ** 2 + y_shifts_in_microns ** 2))

            # Insert
            self.insert1(key)

    def save_video(self, filename='galvo_corrections.mp4', field=1, channel=1,
                   start_index=0, seconds=30, dpi=250):
        """ Creates an animation video showing the original vs corrected scan.

        :param string filename: Output filename (path + filename)
        :param int field: Slice to use for plotting (key for GalvoMotion). Starts at 1
        :param int channel: What channel from the scan to use. Starts at 1
        :param int start_index: Where in the scan to start the video.
        :param int seconds: How long in seconds should the animation run.
        :param int dpi: Dots per inch, controls the quality of the video.

        :returns Figure. You can call show() on it.
        :rtype: matplotlib.figure.Figure
        """
        # Get fps and total_num_frames
        fps = (ScanInfo() & self).fetch1['fps']
        num_video_frames = int(round(fps * seconds))
        stop_index = start_index + num_video_frames

        # Load the scan
        scan_filename = (experiment.Scan() & self).local_filenames_as_wildcard
        scan = scanreader.read_scan(scan_filename)
        scan = scan[field - 1, :, :, channel - 1, start_index: stop_index]
        original_scan = scan.copy()

        # Correct the scan
        correct_motion = (self & {'slice': field}).get_correct_motion()
        correct_raster = (RasterCorrection() & self).get_correct_raster()
        raster_corrected = correct_raster(scan)
        motion_corrected = correct_motion(raster_corrected, slice(start_index, stop_index))
        corrected_scan = motion_corrected

        # Create animation
        import matplotlib.animation as animation

        ## Set the figure
        fig = plt.figure()

        plt.subplot(1, 2, 1)
        plt.title('Original')
        im1 = plt.imshow(original_scan[:, :, 0], vmin=original_scan.min(),
                         vmax=original_scan.max())  # just a placeholder
        plt.axis('off')
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.title('Corrected')
        im2 = plt.imshow(corrected_scan[:, :, 0], vmin=corrected_scan.min(),
                         vmax=corrected_scan.max())  # just a placeholder
        plt.axis('off')
        plt.colorbar()

        ## Make the animation
        def update_img(i):
            im1.set_data(original_scan[:, :, i])
            im2.set_data(corrected_scan[:, :, i])

        video = animation.FuncAnimation(fig, update_img, corrected_scan.shape[2],
                                        interval=1000 / fps)

        # Save animation
        if not filename.endswith('.mp4'):
            filename += '.mp4'
        print('Saving video at:', filename)
        print('If this takes too long, stop it and call again with dpi <', dpi, '(default)')
        video.save(filename, dpi=dpi)

        return fig


@schema
class AverageFrame(dj.Computed):
    definition = """   # average frame for each slice and channel after corrections
    -> MotionCorrection
    -> Channel
    ---
    frame  : longblob     # average frame after Anscombe, max-weighting,
    """

    def _make_tuples(self, key):
        # Read the scan
        scan_filename = (experiment.Scan() & key).local_filenames_as_wildcard
        scan = scanreader.read_scan(scan_filename)

        p = 6  # used for the weighted average

        # Get raster correcting function
        correct_raster = (RasterCorrection() & key).get_correct_raster()

        for field_id in range(scan.num_fields):
            for channel_id in range(scan.num_channels):
                new_tuple = key.copy()
                new_tuple['channel'] = channel_id + 1
                new_tuple['slice'] = field_id + 1

                # Get motion correction function
                galvomotion_rel = (MotionCorrection() & key & {'slice': field_id + 1})
                correct_motion = galvomotion_rel.get_correct_motion()

                # Correct field
                field = scan[field_id, :, :, channel_id, :]
                field = correct_motion(correct_raster(field))

                # l-p norm of each pixel over time
                field[field < 0] = 0
                new_tuple['frame'] = np.mean(field ** p, axis=-1) ** (1 / p)

                # Insert new tuple
                self.insert1(new_tuple)


@schema
class Segmentation(dj.Imported):
    definition = """
    # Correction, source extraction and trace deconvolution of a two-photon scan
    
    -> MotionCorrection
    -> SegmentationMethod
    -> Channel
    """

    @property
    def key_source(self):
        return MotionCorrection() * SegmentationMethod() & dj.OrList(
            [(MotionCorrection() * SegmentationMethod() - 'segmentation="manual"'),
             MotionCorrection() * SegmentationMethod() * ManualSegment()])


    class Trace(dj.Part):
        definition = """  # final calcium trace but before spike extraction or filtering
        -> Segmentation
        trace_id              : smallint
        ---
        trace                 : longblob                     # leave null same as ExtractRaw.Trace
        """


    class Mask(dj.Part):
        definition = """
        # Region of interest produced by segmentation
        
        -> Segmentation.Trace
        ---
        mask_pixels          : longblob      # indices into the image in column major (Fortran) order
        mask_weights = null  : longblob      # weights of the mask at the indices above
        """

        @staticmethod
        def reshape_masks(mask_pixels, mask_weights, px_height, px_width):
            ret = np.zeros((px_height, px_width, len(mask_pixels)))
            for i, (mp, mw) in enumerate(zip(mask_pixels, mask_weights)):
                mask = np.zeros(px_height * px_width)
                mask[mp.squeeze().astype(int) - 1] = mw.squeeze()
                ret[..., i] = mask.reshape(px_height, px_width, order='F')
            return ret

        def get_mask_as_image(self):
            """Return the mask for this single ROI as  an image (2-d array)"""
            # Get params
            pixel_indices, weights, image_height, image_width = \
                (ScanInfo() & self).fetch1['mask_pixels', 'mask_weights', 'px_height', 'px_width']
            # Calculate and reshape mask
            mask_as_vector = np.zeros(image_height * image_width)
            mask_as_vector[pixel_indices - 1] = weights
            spatial_mask = mask_as_vector.reshape(image_height, image_width, order='F')

            return spatial_mask



    class CorrelationImage(dj.Part):
        definition = """
        # Each pixel shows the (average) temporal correlation between that pixel and its eight neighbors
        -> Segmentation
        ---
        correlation_image   : longblob # correlation image
        """

    class BackgroundComponents(dj.Part):
        definition = """
        # Inferred background components with the CNMF algorithm
        
        -> Segmentation
        ---
        masks    : longblob # array (im_width x im_height x num_background_components)
        activity : longblob # array (num_background_components x timesteps)
        """

    class ARCoefficients(dj.Part):
        definition = """
        # Fitted parameters for the autoregressive process (CNMF)
        -> Segmentation.Trace
        ---
        g: longblob # array with g1, g2, ... values for the AR process
        """

    class CNMFParameters(dj.Part):
        definition = """
        # Arguments used to demix and deconvolve the scan with CNMF
        
        -> Segmentation
        
        ---
        
        num_components  : smallint      # estimated number of components
        ar_order        : tinyint       # order of the autoregressive process for impulse function response
        merge_threshold : float         # overlapping masks are merged if temporal correlation greater than this
        num_processes = null    : smallint # number of processes to run in parallel, null=all available
        num_pixels_per_process  : int   # number of pixels processed at a time
        block_size      : int # number of pixels per each dot product
        init_method     : enum("greedy_roi", "sparse_nmf", "local_nmf") # type of initialization used
        soma_radius_in_pixels = null :blob # estimated radius for a soma in the scan
        snmf_alpha = null       : float # regularization parameter for SNMF
        num_background_components : smallint # estimated number of background components
        init_on_patches         : boolean  # whether to run initialization on small patches
        patch_downsampling_factor = null : tinyint # how to downsample the scan
        percentage_of_patch_overlap = null : float # overlap between adjacent patches
        """

    def _make_tuples(self, key):
        """ Load scan one slice and channel at a time, correct for raster and motion
        artifacts and use CNMF to extract sources and deconvolve spike traces.

        See caiman_interface.demix_and_deconvolve_with_cnmf for an explanation of params
        """
        from .utils import caiman_interface as cmn

        print('')
        print('*' * 80)
        print('Processing scan {}'.format(key))
        print('*' * 80)

        # Insert key in ExtractRaw
        self.insert1(key)

        # Read the scan
        scan_filename = (experiment.Scan() & key).local_filenames_as_wildcard
        scan = scanreader.read_scan(scan_filename)

        # Estimate number of components per slice
        num_components = (ScanInfo() & key).estimate_num_components_per_slice()
        num_components += int(round(0.2 * num_components))  # add 20% more just to be sure

        # Estimate the radius of a neuron in the scan (used for somatic scans)
        soma_radius_in_pixels = (ScanInfo() & key).estimate_soma_radius_in_pixels()

        # Set general parameters
        kwargs = {}
        kwargs['num_components'] = num_components
        kwargs['AR_order'] = 2  # impulse response modelling with AR(2) process
        kwargs['merge_threshold'] = 0.8

        # Set performance/execution parameters (heuristically), decrease if memory overflows
        kwargs['num_processes'] = 20  # Set to None for all cores available
        kwargs['num_pixels_per_process'] = 5000
        kwargs['block_size'] = 5000

        # Set params specific to somatic or axonal/dendritic scans
        is_somatic = not (experiment.Session.TargetStructure() & key)
        if is_somatic:
            kwargs['init_method'] = 'greedy_roi'
            kwargs['soma_radius_in_pixels'] = soma_radius_in_pixels
            kwargs['num_background_components'] = 4
            kwargs['init_on_patches'] = False
        else:
            kwargs['init_method'] = 'sparse_nmf'
            kwargs['snmf_alpha'] = 500  # 10^2 to 10^3.5 is a good range
            kwargs['num_background_components'] = 1
            kwargs['init_on_patches'] = True

        # Set params specific to initialization on patches
        if kwargs['init_on_patches']:
            kwargs['patch_downsampling_factor'] = 4
            kwargs['percentage_of_patch_overlap'] = .2

        # Over each channel
        for channel in range(scan.num_channels):
            current_trace_id = 1  # to count traces over one channel, ids start at 1

            # Over each slice in the channel
            for slice in range(scan.num_fields):
                # Load the scan
                print('Loading scan...')
                field = scan[slice, :, :, channel, :]

                # Correct scan
                print('Correcting scan...')
                correct_motion = (MotionCorrection() & key & {'slice': slice + 1}).get_correct_motion()
                correct_raster = (RasterCorrection() & key).get_correct_raster()
                corrected_scan = correct_motion(correct_raster(field))

                # Compute and insert correlation image
                print('Computing correlation image...')
                correlation_image = cmn.compute_correlation_image(corrected_scan)
                Segmentation.CorrelationImage().insert1({**key, 'slice': slice + 1,
                                                         'channel': channel + 1,
                                                         'correlation_image': correlation_image})

                # Extract traces
                print('Extracting mask, traces and spikes (cnmf)...')
                cnmf_result = cmn.demix_and_deconvolve_with_cnmf(corrected_scan, **kwargs)
                (location_matrix, activity_matrix, background_location_matrix,
                 background_activity_matrix, raw_traces, spikes, AR_params) = cnmf_result

                # Obtain new mask order based on their brightness in the correlation image
                new_order = cmn.order_components(location_matrix, correlation_image)

                # Insert traces, spikes and spatial masks (preserving new order)
                print('Inserting masks, traces, spikes, ar parameters and background'
                      ' components...')
                dj.conn().is_connected  # make sure connection is active
                for i in new_order:
                    # Create new trace key
                    trace_key = {**key, 'trace_id': current_trace_id, 'channel': channel + 1}

                    # Insert traces and spikes
                    Segmentation.Trace().insert1({**trace_key, 'raw_trace': raw_traces[i, :]})
                    Segmentation.SpikeRate().insert1({**trace_key, 'spike_trace': spikes[i, :]})

                    # Insert fitted AR parameters
                    if kwargs['AR_order'] > 0:
                        Segmentation.ARCoefficients().insert1({**trace_key, 'g': AR_params[i, :]})

                    # Get indices and weights of defined pixels in mask (matlab-like)
                    mask_as_F_ordered_vector = location_matrix[:, :, i].ravel(order='F')
                    defined_mask_indices = np.where(mask_as_F_ordered_vector)[0]
                    defined_mask_weights = mask_as_F_ordered_vector[defined_mask_indices]
                    defined_mask_indices += 1  # matlab indices start at 1

                    # Insert spatial mask
                    # TODO: Channel?
                    Segmentation.Mask().insert1({**trace_key, 'slice': slice + 1,
                                                 'mask_pixels': defined_mask_indices,
                                                 'mask_weights': defined_mask_weights})

                    # Increase trace_id counter
                    current_trace_id += 1

                # Insert background components
                background_dict = {**key, 'channel': channel + 1, 'slice': slice + 1,
                                   'masks': background_location_matrix,
                                   'activity': background_activity_matrix}
                Segmentation.BackgroundComponents().insert1(background_dict)

        # Insert CNMF parameters (one per scan)
        lowercase_kwargs = {key.lower(): value for key, value in kwargs.items()}
        Segmentation.CNMFParameters().insert1({**key, **lowercase_kwargs})

    def save_video(self, filename='cnmf_results.mp4', field=1, channel=1,
                   start_index=0, seconds=30, dpi=250, first_n=None):
        """ Creates an animation video showing the original vs corrected scan.

        :param string filename: Output filename (path + filename)
        :param int field: Slice to use for plotting. Starts at 1
        :param int channel: What channel from the scan to use. Starts at 1
        :param int start_index: Where in the scan to start the video.
        :param int seconds: How long in seconds should the animation run.
        :param int dpi: Dots per inch, controls the quality of the video.
        :param int first_n: Consider only the first n components.

        :returns Figure. You can call show() on it.
        :rtype: matplotlib.figure.Figure
        """
        # Get fps and calculate total number of frames
        fps = (ScanInfo() & self).fetch1['fps']
        num_video_frames = int(round(fps * seconds))
        stop_index = start_index + num_video_frames

        # Load the scan
        scan_filename = (experiment.Scan() & self).local_filenames_as_wildcard
        scan = scanreader.read_scan(scan_filename)
        scan = scan[field - 1, :, :, channel - 1, start_index: stop_index]

        # Correct the scan
        correct_motion = (MotionCorrection() & self & {'slice': field}).get_correct_motion()
        correct_raster = (RasterCorrection() & self).get_correct_raster()
        raster_corrected = correct_raster(scan)
        motion_corrected = correct_motion(raster_corrected, slice(start_index, stop_index))
        scan = motion_corrected

        # Get scan dimensions
        image_height, image_width, _ = scan.shape
        num_pixels = image_height * image_width

        # Get location and activity matrices
        location_matrix = self.get_all_masks(field, channel)
        activity_matrix = self.get_all_traces(field, channel)
        background_rel = Segmentation.BackgroundComponents() & self & {'slice': field,
                                                                       'channel': channel}
        background_location_matrix, background_activity_matrix = \
            background_rel.fetch1['masks', 'activity']

        # Select first n components
        if first_n is not None:
            location_matrix = location_matrix[:, :, :first_n]
            activity_matrix = activity_matrix[:first_n, :]

        # Restrict computations to the necessary video frames
        activity_matrix = activity_matrix[:, start_index: stop_index]
        background_activity_matrix = background_activity_matrix[:, start_index: stop_index]

        # Calculate matrices
        extracted = np.dot(location_matrix.reshape(num_pixels, -1), activity_matrix)
        extracted = extracted.reshape(image_height, image_width, -1)
        background = np.dot(background_location_matrix.reshape(num_pixels, -1),
                            background_activity_matrix)
        background = background.reshape(image_height, image_width, -1)
        residual = scan - extracted - background

        # Create animation
        import matplotlib.animation as animation

        ## Set the figure
        fig = plt.figure()

        plt.subplot(2, 2, 1)
        plt.title('Original (Y)')
        im1 = plt.imshow(scan[:, :, 0], vmin=scan.min(), vmax=scan.max())  # just a placeholder
        plt.axis('off')
        plt.colorbar()

        plt.subplot(2, 2, 2)
        plt.title('Extracted (A*C)')
        im2 = plt.imshow(extracted[:, :, 0], vmin=extracted.min(), vmax=extracted.max())
        plt.axis('off')
        plt.colorbar()

        plt.subplot(2, 2, 3)
        plt.title('Background (B*F)')
        im3 = plt.imshow(background[:, :, 0], vmin=background.min(),
                         vmax=background.max())
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
            im2.set_data(extracted[:, :, i])
            im3.set_data(background[:, :, i])
            im4.set_data(residual[:, :, i])

        video = animation.FuncAnimation(fig, update_img, scan.shape[2],
                                        interval=1000 / fps)

        # Save animation
        if not filename.endswith('.mp4'):
            filename += '.mp4'
        print('Saving video at:', filename)
        print('If this takes too long, stop it and call again with dpi <', dpi, '(default)')
        video.save(filename, dpi=dpi)

        return fig

    def plot_contours(self, slice=1, channel=1, first_n=None):
        """ Draw contours of masks over the correlation image.

        :param slice: Scan slice to use
        :param channel: Scan channel to use
        :param first_n: Number of masks to plot. None for all.
        :returns: None
        """
        from .utils import caiman_interface as cmn

        # Get location matrix
        location_matrix = self.get_all_masks(slice, channel)

        # Select first n components
        if first_n is not None:
            location_matrix = location_matrix[:, :, :first_n]

        # Get correlation image if defined
        image_rel = Segmentation.CorrelationImage() & self & {'slice': slice,
                                                              'channel': channel}
        correlation_image = image_rel.fetch1['correlation_image'] if image_rel else None

        # Draw contours
        cmn.plot_contours(location_matrix, correlation_image)

    def plot_centroids(self, slice=1, channel=1, first_n=None):
        """ Draw centroids of masks over the correlation image.

        :param slice: Scan slice to use
        :param channel: Scan channel to use
        :param first_n: Number of masks to plot. None for all.
        :returns: None
        """
        from .utils import caiman_interface as cmn

        # Get location matrix
        location_matrix = self.get_all_masks(slice, channel)

        # Select first n components
        if first_n is not None:
            location_matrix = location_matrix[:, :, :first_n]

        # Get correlation image if defined
        image_rel = Segmentation.CorrelationImage() & self & {'slice': slice,
                                                              'channel': channel}
        correlation_image = image_rel.fetch1['correlation_image'] if image_rel else None

        # Draw centroids
        cmn.plot_centroids(location_matrix, correlation_image)

    def plot_impulse_responses(self, slice=1, channel=1, num_timepoints=100):
        """ Plots the individual impulse response functions for all traces assuming an
        autoregressive process (p > 0).

        :param int num_timepoints: The number of points after impulse to use for plotting.

        :returns Figure. You can call show() on it.
        :rtype: matplotlib.figure.Figure
        """
        ar_rel = Segmentation.ARCoefficients() & self & {'slice': slice, 'channel': channel}
        fps = (ScanInfo() & self).fetch1['fps']

        # Get AR coefficients
        ar_coefficients = ar_rel.fetch['g'] if ar_rel else None

        if ar_coefficients is not None:
            fig = plt.figure()
            x_axis = np.arange(num_timepoints) / fps  # make it seconds

            # Over each trace
            for g in ar_coefficients:
                AR_order = len(g)

                # Calculate impulse response function
                irf = np.zeros(num_timepoints)
                irf[0] = 1  # initial spike
                for i in range(1, num_timepoints):
                    if i <= AR_order:  # start of the array needs special care
                        irf[i] = np.sum(g[:i] * irf[i - 1:: -1])
                    else:
                        irf[i] = np.sum(g * irf[i - 1: i - AR_order - 1: -1])

                # Plot
                plt.plot(x_axis, irf)

            return fig

    def get_all_masks(self, slice, channel):
        """Returns an image_height x image_width x num_masks matrix with all masks."""
        mask_rel = Segmentation.Mask() & self & {'slice': slice, 'channel': channel}

        # Get masks
        image_height, image_width = (ScanInfo() & self).fetch1['px_height',
                                                               'px_width']
        mask_pixels, mask_weights = mask_rel.fetch.order_by('trace_id')['mask_pixels',
                                                                        'mask_weights']

        # Reshape masks
        location_matrix = Segmentation.Mask.reshape_masks(mask_pixels, mask_weights,
                                                          image_height, image_width)

        return location_matrix

    def get_all_traces(self, slice, channel):
        """ Returns a num_traces x num_timesteps matrix with all traces."""
        trace_rel = Segmentation.Trace() * Segmentation.Mask() & self & {'slice': slice,
                                                                         'channel': channel}
        # Get traces
        raw_traces = trace_rel.fetch.order_by('trace_id')['raw_trace']

        # Reshape traces
        raw_traces = np.array([x.squeeze() for x in raw_traces])

        return raw_traces

    def get_all_spikes(self, slice, channel):
        """ Returns a num_spike_traces x num_timesteps matrix with all spike rates."""
        spike_rel = Segmentation.SpikeRate() * Segmentation.Mask() & self & {'slice': slice,
                                                                             'channel': channel}
        # Get spike traces
        spike_traces = spike_rel.fetch.order_by('trace_id')['spike_trace']

        # Reshape them
        spike_traces = np.array([x.squeeze() for x in spike_traces])

        return spike_traces


@schema
class Sync(dj.Imported):
    definition = """
    -> ScanInfo
    ---
    -> vis.Session
    first_trial                 : int                           # first trial index from vis.Trial overlapping recording
    last_trial                  : int                           # last trial index from vis.Trial overlapping recording
    signal_start_time           : double                        # (s) signal start time on stimulus clock
    signal_duration             : double                        # (s) signal duration on stimulus time
    frame_times = null          : longblob                      # times of frames and slices
    sync_ts=CURRENT_TIMESTAMP   : timestamp                     # automatic
    """




@schema
class Spikes(dj.Computed):
    definition = """  
    # infer spikes from calcium traces
    
    -> Segmentation
    -> SpikeMethod
    """

    @property
    def key_source(self):
        return ((Segmentation() * SpikeMethod() & "language='python'") - 'segmentation="nmf"').proj()

    class RateTrace(dj.Part):
        definition = """  
        # Deconvolved calcium acitivity
        
        -> Spikes
        -> Segmentation.Trace
        ---
        rate_trace: longblob     # leave null same as ExtractRaw.Trace
        """


    def _make_tuples(self, key):
        try:
            import pyfnnd
        except ImportError:
            warn(
                'Could not load pyfnnd. Oopsi spike inference will fail. Install from https://github.com/cajal/PyFNND.git')

        print('Populating Spikes for ', key, end='...', flush=True)
        method = (SpikeMethod() & key).fetch1['spike_method_name']
        if method == 'stm':
            prep = ScanInfo() & key
            fps = prep.fetch1['fps']
            X = [dict(trace=fill_nans(x['trace'].astype('float64'))) for x in
                 (Segmentation.Trace() & key).proj('trace').fetch.as_dict]

            self.insert1(key)
            for x in (SpikeMethod() & key).spike_traces(X, fps):
                self.RateTrace().insert1(dict(key, **x))

        elif method == 'oopsi':
            prep = ScanInfo() & key
            self.insert1(key)
            fps = prep.fetch1['fps']
            part = self.RateTrace()
            for trace, trace_key in zip(*(Segmentation.Trace() & key).fetch['trace', dj.key]):
                trace = pyfnnd.deconvolve(fill_nans(np.float64(trace.flatten())), dt=1 / fps)[0]
                part.insert1(dict(trace_key, rate_trace=trace.astype(np.float32)[:, np.newaxis], **key))
        else:
            raise NotImplementedError('Method {spike_method} not implemented.'.format(**key))
        print('Done', flush=True)


@schema
class BehaviorSync(dj.Imported):
    definition = """
    -> experiment.Scan
    ---
    frame_times                  : longblob # time stamp of imaging frame on behavior clock
    """


schema.spawn_missing_classes()
