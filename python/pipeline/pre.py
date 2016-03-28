import datajoint as dj
from . import exp
schema = dj.schema('pipeline_unified_preprocessing', locals())

# TODO:
# * where does ScanInfo go?
# * where does Channel go? AlignMotion needs that.
# * include Sync


# TODO: define parameters here
@schema
class AODImportParams(dj.Imported):
    definition = """
    # cell segmentation method
    segment_method         : tinyint # id of the method
    -----
    method_name            : char(8) # name of the method for switch statements

    """

    contents = [
        (1, 'manual'), (2, 'nmf')
    ]

@schema
class SegmentMethod(dj.Lookup):
    definition = """

    """

@schema
class ExtractRaw(dj.Imported):
    definition = """
    # grouping table for extracted traces from AOD and Reso Scans
    ->exp.Scan
    ---
    """

    class ResoScanInfo(dj.Part):
        definition = """
        # header information for Reso Scans
        -> ExtractRaw
        ---
        nframes_requested           : int                           # number of valumes (from header)
        nframes                     : int                           # frames recorded
        px_width                    : smallint                      # pixels per line
        px_height                   : smallint                      # lines per frame
        um_width                    : float                         # width in microns
        um_height                   : float                         # height in microns
        bidirectional               : tinyint                       # 1=bidirectional scanning
        fps                         : float                         # (Hz) frames per second
        zoom                        : decimal(4,1)                  # zoom factor
        dwell_time                  : float                         # (us) microseconds per pixel per frame
        nchannels                   : tinyint                       # number of recorded channels
        nslices                     : tinyint                       # number of slices
        slice_pitch                 : float                         # (um) distance between slices
        fill_fraction               : float                         # raster scan fill fraction (see scanimage)
        """

    class Segment(dj.Part):
        definition = """
        -> AlignMotion
        -> SegmentMethod
        ---
        segment_ts=CURRENT_TIMESTAMP: timestamp                     # automatic

        """

    class Unit(dj.Part):
        definition = """
        ->ExtractRaw
        """

    class RawTrace(dj.Part):
        definition = """
        ->ExtractRaw.Unit
        raw_trace_id        : int
        """

    class ImportAOD(dj.Part):
        definition = """
        ->ExtractRaw
        ->AODImportParams
        """

    class Points(dj.Part):
        definition = """
        ->ExtractRaw.ImportAOD
        """

@schema
class AlignRaster(dj.Computed):
    definition = """
    -> ExtractRaw.ScanInfo
    ---
    raster_phase                : float                         # shift of odd vs even raster lines
    """


@schema
class AlignMotion(dj.Computed):
    definition = """
    # motion correction
    -> AlignRaster
    -> exp.Slice
    ---
    -> exp.Channel
    motion_xy                   : longblob                      # (pixels) y,x motion correction offsets
    motion_rms                  : float                         # (um) stdev of motion
    align_times=CURRENT_TIMESTAMP: timestamp                    # automatic
    avg_frame=null              : longblob                      # averaged aligned frame
    INDEX(animal_id,session,scan_idx,channel)
    """

@schema
class ScanROI(dj.Computed):
    definition = """
    # mask of a segmented cell
    -> ExtractRaw.Segment
    scan_roi_id                 : smallint # id of the mask
    -----
    roi_pixels                  : longblob # indices into the image in column major (Fortran) order
    roi_weights                 : longblob # weights of the mask at the indices above
    """

@schema
class ComputeTraces(dj.Computed):
    definition = """
    ->ExtractRaw

    """

    class Trace(dj.Part):
        definition = """
        ->ComputeTraces
        ->ExtractRaw.Unit
        """