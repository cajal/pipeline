from . import schema
import datajoint as dj 
from pipeline import experiment
import json 
import scanreader
from ..shared import Field, Channel

CURRENT_VERSION = 1

class CorrectionChannel(dj.Manual):
    definition = """ # channel to use for raster and motion correction

    -> experiment.Scan
    -> shared.Field
    ---
    -> shared.Channel
    """

    def fill(self, key, channel=1):
        for field_key in (ScanInfo.Field() & key).fetch(dj.key):
            self.insert1(
                {**field_key, "channel": channel},
                ignore_extra_fields=True,
                skip_duplicates=True,
            )

@schema
class Version(dj.Manual):
    definition = """ # versions for the meso pipeline

    -> shared.PipelineVersion
    ---
    description = ''                : varchar(256)      # any notes on this version
    date = CURRENT_TIMESTAMP        : timestamp         # automatic
    """


@schema
class ScanInfo(dj.Imported):
    definition = """ # general data about mesoscope scans

    -> experiment.Scan
    -> Version                                  # meso version
    ---
    nfields                 : tinyint           # number of fields
    nchannels               : tinyint           # number of channels
    nframes                 : int               # number of recorded frames
    nframes_requested       : int               # number of requested frames (from header)
    x                       : float             # (um) ScanImage's 0 point in the motor coordinate system
    y                       : float             # (um) ScanImage's 0 point in the motor coordinate system
    fps                     : float             # (Hz) frames per second
    bidirectional           : boolean           # true = bidirectional scanning
    usecs_per_line          : float             # microseconds per scan line
    fill_fraction           : float             # raster scan temporal fill fraction (see scanimage)
    nrois                   : tinyint           # number of ROIs (see scanimage)
    """

    @property
    def key_source(self):
        rigs = [{"rig": "2P4"}, {"rig": "R2P1"}]
        meso_scans = experiment.Scan() & (experiment.Session() & rigs)
        return meso_scans * (Version() & {"pipe_version": CURRENT_VERSION})

    class Field(dj.Part):
        definition = """ # field-specific scan information

        -> ScanInfo
        -> shared.Field
        ---
        px_height           : smallint      # height in pixels
        px_width            : smallint      # width in pixels
        um_height           : float         # height in microns
        um_width            : float         # width in microns
        x                   : float         # (um) center of field in the motor coordinate system
        y                   : float         # (um) center of field in the motor coordinate system
        z                   : float         # (um) absolute depth with respect to the surface of the cortex
        delay_image         : longblob      # (ms) delay between the start of the scan and pixels in this field
        roi                 : tinyint       # ROI to which this field belongs
        valid_depth=false   : boolean       # whether depth has been manually check
        """

        def make(self, key, scan, field_id):
            # Create results tuple
            tuple_ = key.copy()
            tuple_["field"] = field_id + 1

            # Get attributes
            (
                x_zero,
                y_zero,
                _,
            ) = scan.motor_position_at_zero  # motor x, y at ScanImage's 0
            surf_z = (experiment.Scan() & key).fetch1(
                "depth"
            )  # surface depth in fastZ coordinates
            tuple_["px_height"] = scan.field_heights[field_id]
            tuple_["px_width"] = scan.field_widths[field_id]
            tuple_["um_height"] = scan.field_heights_in_microns[field_id]
            tuple_["um_width"] = scan.field_widths_in_microns[field_id]
            tuple_["x"] = x_zero + scan._degrees_to_microns(scan.fields[field_id].x)
            tuple_["y"] = y_zero + scan._degrees_to_microns(scan.fields[field_id].y)
            tuple_["z"] = scan.field_depths[field_id] - surf_z  # fastZ only
            tuple_["delay_image"] = scan.field_offsets[field_id]
            tuple_["roi"] = scan.field_rois[field_id][0]

            # Insert
            self.insert1(tuple_)

        @property
        def microns_per_pixel(self):
            """ Returns an array with microns per pixel in height and width. """
            um_height, px_height, um_width, px_width = self.fetch1(
                "um_height", "px_height", "um_width", "px_width"
            )
            return np.array([um_height / px_height, um_width / px_width])

    def make(self, key):
        """ Read and store some scan parameters."""
        # Read the scan
        print("Reading header...")
        scan_filename = (experiment.Scan() & key).local_filenames_as_wildcard
        scan = scanreader.read_scan(scan_filename)


        # Get attributes
        tuple_ = key.copy()  # in case key is reused somewhere else
        tuple_["nfields"] = scan.num_fields
        tuple_["nchannels"] = scan.num_channels
        tuple_["nframes"] = scan.num_frames
        tuple_["nframes_requested"] = scan.num_requested_frames
        tuple_["x"] = scan.motor_position_at_zero[0]
        tuple_["y"] = scan.motor_position_at_zero[1]
        tuple_["fps"] = scan.fps
        tuple_["bidirectional"] = scan.is_bidirectional
        tuple_["usecs_per_line"] = scan.seconds_per_line * 1e6
        tuple_["fill_fraction"] = scan.temporal_fill_fraction
        tuple_["nrois"] = scan.num_rois
        tuple_["valid_depth"] = True

        # Insert in ScanInfo
        self.insert1(tuple_)

        # Insert field information
        for field_id in range(scan.num_fields):
            ScanInfo.Field().make(key, scan, field_id)

        # Fill in CorrectionChannel if only one channel
        if scan.num_channels == 1:
            CorrectionChannel().fill(key)

        # Fill SegmentationTask if scan in autosegment
        if experiment.AutoProcessing() & key & {"autosegment": True}:
            SegmentationTask().fill(key)