from warnings import warn
import datajoint as dj

schema = dj.schema('pipeline_shared', locals())

@schema
class Slice(dj.Lookup):
    definition = """  # slices in resonant scans
    slice       : tinyint
    """
    contents = [[i] for i in range(1, 13)]


@schema
class Channel(dj.Lookup):
    definition = """  # recording channel, directly related to experiment.PMTFilterSet.Channel
    channel     : tinyint
    """
    contents = [[i] for i in range(1, 5)]

@schema
class SegmentationMethod(dj.Lookup):
    definition = """
    #  methods for trace extraction from raw data for either AOD or Galvo data

    extract_method      : tinyint
    ---
    segmentation        : varchar(16)
    """

    contents = zip([1, 2], ['manual', 'nmf'])

@schema
class MaskType(dj.Lookup):
    definition = """ # possible classifications for a segmented mask
    type        : varchar(16)
    """
    contents = [
        ['soma'],
        ['axon'],
        ['dendrite'],
        ['neuropil'],
        ['artifact'],
        ['unknown']
    ]

@schema
class SpikeMethod(dj.Lookup):
    definition = """
    spike_method            : smallint              # spike inference method
    ---
    spike_method_name       : varchar(16)           # short name to identify the spike inference method
    spike_method_details    : varchar(255)          # more details
    language                : enum('matlab', 'python')   # implementation language
    """

    contents = [
        [2, "oopsi", "nonnegative sparse deconvolution from Vogelstein (2010)", "python"],
        [3, "stm", "spike triggered mixture model from Theis et al. (2016)", "python"],
        [5, "nmf", "noise constrained deconvolution from Pnevmatikakis et al., 2016", "python"]
    ]