import datajoint as dj
from . import experiment, psy

from distutils.version import StrictVersion
assert StrictVersion(dj.__version__) >= StrictVersion('0.2.8')

schema = dj.schema('pipeline_preprocess', locals())


@schema
class Slice(dj.Lookup):
    definition = """  # slices in resonant scanner scans
    slice  : tinyint  # slice in scan
    """
    contents = ((i,) for i in range(12))


@schema
class AodImportParam(dj.Lookup):
    definition = """ # options for importing AOD scans
    aod_param_opt: tinyint
    ---
    """


@schema
class SegmentMethod(dj.Lookup):
    definition = """
    segment_method : tinyint  # calcium image segmentation method
    ---
    segment_method_name : varchar(30)    #
    """
    contents = [[1, 'manual'], [2, 'nmf']]


@schema
class Gather(dj.Imported):
    definition = """  # master table that gathers data about the scans of different types, prepares for trace extraction
    -> experiment.Scan
    """

    class Galvo(dj.Part):
        definition = """    # basic information about resonant microscope scans, raster correction
        -> Gather
        ---
        nframes_requested       : int               # number of valumes (from header)
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
        fill_fraction           : float             # raster scan fill fraction (see scanimage)
        raster_phase            : float             # shift of odd vs even raster lines
        avg_frame               : longblob          # raw average frame
        min_intensity           : int               # min value in movie
        max_intensity           : int               # max value in movie
        mean_diff_squared_hist  : longblob          # measured frame-to-frame variance for each intensity bin
        """

    class GalvoMotion(dj.Part):
        definition = """   # motion correction for galvo scans
        -> Gather.Galvo
        -> Slice
        ---
        -> experiment.PMTFilterSet.Channel
        motion_xy                   : longblob       # (pixels) y,x motion correction offsets
        motion_rms                  : float          # (um) stdev of motion
        align_times=CURRENT_TIMESTAMP: timestamp     # automatic
        """

    class Aod(dj.Part):
        definition = """   # information about AOD scans
        -> Gather
        ---
        -> AodImportParam
        """

    class AodPoint(dj.Part):
        definition = """
        -> Gather.Aod
        point_id : smallint    # id of a scan point
        ---
        x: float   # (um)
        y: float   # (um)
        z: float   # (um)
        """


@schema
class ExtractRaw(dj.Imported):
    definition = """  # pre-processing of a twp-photon scan
    -> Gather
    ----
    """

    class Trace(dj.Part):
        definition = """  # raw trace, common to Galvo
        -> ExtractRaw
        -> experiment.PMTFilterSet.Channel
        trace_id  : smallint
        ---
        raw_trace : longblob     # unprocessed calcium trace
        """

    class GalvoSegmentation(dj.Part):
        definition = """  # segmentation of galvo movies
        -> ExtractRaw
        -> SegmentMethod
        ---
        segmentation_mask=null  :  longblob
        """

    class GalvoROI(dj.Part):
        definition = """  # region of interest produced by segmentation
        -> ExtractRaw.GalvoSegmentation
        -> ExtractRaw.Trace
        ---
        mask_pixels          :longblob      # indices into the image in column major (Fortran) order
        mask_weights = null  :longblob      # weights of the mask at the indices above
        """

    def _make_tuples(self, key):
        """ implemented in matlab """
        raise NotImplementedError


@schema
class ComputeTraces(dj.Computed):
    definition = """   # compute traces
    -> ExtractRaw
    ---
    """

    class Trace(dj.Part):
        definition = """  # final calcium trace but before spike extraction or filtering
        -> ComputeTraces
        -> ExtractRaw.Trace
        ---
        trace = null  : longblob     # leave null same as ExtractRaw.Trace
        """

    def _make_tuples(self, key):
        raise NotImplementedError


@schema
class Sync(dj.Imported):
    definition = """
    -> Gather
    ---
    -> psy.Session
    first_trial                 : int                           # first trial index from psy.Trial overlapping recording
    last_trial                  : int                           # last trial index from psy.Trial overlapping recording
    signal_start_time           : double                        # (s) signal start time on stimulus clock
    signal_duration             : double                        # (s) signal duration on stimulus time
    frame_times = null          : longblob                      # times of frames and slices
    sync_ts=CURRENT_TIMESTAMP   : timestamp                     # automatic
    """

