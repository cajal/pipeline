""" Lookup tables shared among multi-photon pipelines. """

import datajoint as dj

schema = dj.schema('pipeline_shared', locals(), create_tables=False)

@schema
class Field(dj.Lookup):
    definition = """ # fields in mesoscope scans
    field       : tinyint
    """
    contents = [[i] for i in range(1, 25)]

@schema
class Channel(dj.Lookup):
    definition = """  # recording channel, directly related to experiment.PMTFilterSet.Channel
    channel     : tinyint
    """
    contents = [[i] for i in range(1, 5)]

@schema
class PipelineVersion(dj.Lookup):
    definition = """ # versions for the reso pipeline

    pipe_version                    : smallint
    """
    contents = [[i] for i in range(3)]

@schema
class SegmentationMethod(dj.Lookup):
    definition = """ # methods for mask extraction for multi-photon scans
    segmentation_method         : tinyint
    ---
    name                        : varchar(16)
    details                     : varchar(255)
    language                    : enum('matlab', 'python')  # implementation language
    """
    contents = [
        [1, 'manual', '', 'matlab'],
        [2, 'nmf', 'constrained non-negative matrix factorization from Pnevmatikakis et al. (2016)',
         'python'],
        [3, 'nmf-patches', 'same as nmf but initialized in small image patches', 'python'],
        [4, 'nmf-boutons', 'nmf for axonal terminals', 'python'],
        [5, '3d-conv', 'masks from the segmentation of the stack', 'python']
    ]

@schema
class StackSegmMethod(dj.Lookup):
    definition = """ # methods for 3-d stack segmentations
    stacksegm_method         : tinyint
    ---
    name                        : varchar(16)
    details                     : varchar(255)
    language                    : enum('matlab', 'python')  # implementation language
    """
    contents = [
        [1, '3dconv', '3-d convolutional network plus watershed', 'python'],
        [2, '3dconv-ensemble', 'an ensemble of 3-d convolutional networks plus watershed', 'python']
    ]

@schema
class ClassificationMethod(dj.Lookup):
    definition = """ # methods to classify extracted masks
    classification_method         : tinyint
    ---
    name                        : varchar(16)
    details                     : varchar(255)
    language                    : enum('matlab', 'python')  # implementation language
    """

    contents = [
    [1, 'manual', 'masks classified by visual inspection', 'python'],
    [2, 'cnn-caiman', 'classification made by a trained convolutional network', 'python']
    ]

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
    spike_method        : tinyint                   # spike inference method
    ---
    name                : varchar(16)               # short name to identify the spike inference method
    details             : varchar(255)              # more details
    language            : enum('matlab', 'python')  # implementation language
    """

    contents = [
        [2, 'foopsi', 'nonnegative sparse deconvolution from Vogelstein (2010)', 'python'],
        [3, 'stm', 'spike triggered mixture model from Theis et al. (2016)', 'python'],
        [5, 'nmf', 'noise constrained deconvolution from Pnevmatikakis et al. (2016)', 'python']
    ]

@schema
class RegistrationMethod(dj.Lookup):
    definition = """
    registration_method : tinyint                   # method used for registration
    ---
    name                : varchar(16)               # short name to identify the registration method
    details             : varchar(255)              # more details
    language            : enum('matlab', 'python')  # implementation language
    """

    contents = [
        [1, 'rigid', '3-d cross-correlation (40 microns above and below estimated z)', 'python'],
        [2, 'rigid2', '3-d cross-correlation (100 microns above and below estimated z)', 'python'],
        [3, 'affine', ('exhaustive search of 3-d rotations + cross-correlation (40 microns'
                       'above and below estimated z)'), 'python'],
        [4, 'affine2', ('exhaustive search of 3-d rotations + cross-correlation (100 microns'
                        'above and below estimated z)'), 'python']
    ]

@schema
class CurationMethod(dj.Lookup):
    definition = """
    curation_method     : tinyint                   # method to curate the initial registration estimates
    ---
    name                : varchar(16)               # short name to identify the curation method
    details             : varchar(255)              # more details
    language            : enum('matlab', 'python')  # implementation language
    """

    contents = [
        [1, 'none', 'estimates are left unchanged', 'python'],
        [2, 'manual', 'manually inspect each field estimate', 'matlab'],
    ]