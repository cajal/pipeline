import datajoint as dj
import experiment
import psy

from distutils.version import StrictVersion
assert StrictVersion(dj.__version__) >= StrictVersion('0.2.5')
schema = dj.schema('unipipe_preprocessing', locals())

@schema
class Slice(dj.Lookup):
    definition = """
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
class SegmentationMethod(dj.Lookup):
    definition = """
    segment_method :tinyint  # calcium image segmentation method
    ---
    description : varchar(255)    #
    """

@schema
class ExtractRaw(dj.Imported):
    definition = """  # pre-processing of a twp-photon scan
    -> expe.Scan
    ----
    """

    class Sync(dj.Part):
        definition = """
        -> ExtractRaw
        ---
        -> psy.Session
        """

    class Unit(dj.Part):
        definition = """
        -> ExtractRaw
        unit_id  : smallint
        """

    # --------  AOD  ---------- #

    class ImportAod(dj.Part):
        definition = """   # information about AOD scans
        -> ExtractRaw
        ---
        -> AodImportParam
        """

    class Point(dj.Part):
        definition = """
        -> ExtractRaw.ImportAod
        -> ExtractRaw.Unit
        ---
        x: float   # (um)
        y: float   # (um)
        z: float   # (um)
        """

    # --------- Resonant galvo --------- #
    class ScanInfo(dj.Part):
        definition = """  # basic information about resonant microscope scans
        -> ExtractRaw
        ---
        """

    class AlignMotion(dj.Part):
        definition = """
        -> ExtractRaw.ScanInfo
        -> Slice
        ---
        """

    class Segmentation(dj.Part):
        definition = """
        -> ExtractRaw.AlignMotion
        -> SegmentationMethod
        ---
        segmentation_mask  :  longblob
        """

    class ROI(dj.Part):
        definition = """
        -> ExtractRaw.Segmentation
        -> ExtractRaw.Unit
        ---
        roi_pixels  : longblob   # pixel indices
        """

    # ------ common to both AOD and Reso ------ #

    class RawTrace(dj.Part):
         definition = """
         -> ExtractRaw.Unit
         -> expe.PMTChannel
         ---
         raw_trace : longblob     # unprocessed calcium trace
         """

    def _make_tuples(self, key):
        raise NotImplementedError


@schema
class ComputeTraces(dj.Computed):
    definition = """
    -> ExtractRaw
    ---
    """

    class Trace(dj.Part):
        definition = """
        -> ComputeTraces
        ---
        trace_id : smallint
        """

    def _make_tuples(self, key):
        raise NotImplementedError
