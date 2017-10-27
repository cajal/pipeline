import sys
import datajoint as dj
from distutils.version import StrictVersion
from . import experiment, reso, meso, shared
from .exceptions import PipelineException


schema = dj.schema('pipeline_fuse', locals())


@schema
class Pipe(dj.Lookup):
    definition = """
    # Names of fused pipelines
    pipe  : varchar(16)    # pipeline name
    """
    contents = zip(('reso', 'meso'))


class Resolver:
    """
    This mixin class populates tables that fuse two pipelines
    """

    def make(self, key):
        # find the matching pipeline from those specified in self.mapping
        try:
            pipe, src, dest = next(((pipe, src, dest)
                for pipe, (src, dest) in self.mapping.items() if src() & key))
        except StopIteration:
            raise PipelineException(
                    'The key source yielded a key from an uknown pipeline')
        self.insert1(dict(key, pipe=pipe))
        dest().insert(src() & key, ignore_extra_fields=True)
        return src

    @property
    def module(self):
        """
        (fuse.Activity() & key).module is the module where the activity resides.
        Throws an error if the activity comes from multiple modules.
        Does the same thing for fuse.ScanDone() or any other subclass of Resolver
        """
        pipes = (dj.U('pipe') & self).fetch('pipe')
        if len(pipes) != 1:
            raise PipelineException('Please restrict query to a single pipeline.')
        rel, _ = self.mapping[pipes[0]]
        return sys.modules[rel.__module__]


@schema
class Activity(Resolver, dj.Computed):
    definition = """
    # Calcium activity for a single field within a scan
    -> experiment.Scan
    -> shared.PipelineVersion
    -> shared.Field
    -> shared.Channel
    -> shared.SegmentationMethod
    -> shared.SpikeMethod
    ---
    -> Pipe
    """

    class Trace(dj.Part):
        definition = """
        # Trace corresponding to <module>.Activity.Trace
        -> master
        unit_id  : int   # unique per scan & segmentation method
        """

    @property
    def key_source(self):
        assert StrictVersion(dj.__version__) >= StrictVersion('0.9.0'), "Please upgrade datajoint to version 0.9.0+"
        return meso.Activity().proj() + reso.Activity().proj()

    class Reso(dj.Part):
        definition = """
        -> reso.Activity
        -> Activity
        """

    class Meso(dj.Part):
        definition = """
        -> meso.Activity
        -> Activity
        """

    @property
    def mapping(self):
        return {'meso': (meso.Activity, Activity.Meso),
                'reso': (reso.Activity, Activity.Reso)}


    def make(self, key):
        src = super().make(key)
        module = sys.modules[src.__module__]
        self.Trace().insert(self * module.Activity.Trace() & key, ignore_extra_fields=True)


@schema
class ScanDone(Resolver, dj.Computed):
    definition = """
    # Calcium activity for the whole scan (multiple scan fields)
    -> experiment.Scan
    -> shared.PipelineVersion
    -> shared.SegmentationMethod
    -> shared.SpikeMethod
    ---
    -> Pipe
    """

    @property
    def key_source(self):
        assert StrictVersion(dj.__version__) >= StrictVersion('0.9.0'), "Please upgrade datajoint to version 0.9.0+"
        return meso.ScanDone().proj() + reso.ScanDone().proj()

    class Reso(dj.Part):
        definition = """
        -> reso.ScanDone
        -> ScanDone
        """

    class Meso(dj.Part):
        definition = """
        -> meso.ScanDone
        -> ScanDone
        """

    @property
    def mapping(self):
        return {'meso': (meso.ScanDone, ScanDone.Meso),
                'reso': (reso.ScanDone, ScanDone.Reso)}
