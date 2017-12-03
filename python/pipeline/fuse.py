import sys
import datajoint as dj
from distutils.version import StrictVersion
from . import experiment, reso, meso, shared
from .exceptions import PipelineException


schema = dj.schema('pipeline_fuse', locals())


@schema
class Pipe(dj.Lookup):
    definition = """ # names of fused pipelines
    pipe                : varchar(16)                   # pipeline name
    """
    contents = [['reso'], ['meso']]


class Resolver:
    """ This mixin class populates tables that fuse two pipelines """
    @property
    def module(self):
        """ (fuse.Activity() & key).module is the module where the activity resides.

        Raises an exception if the activity comes from multiple modules. Works similarly
        for other subclasses of Resolver.
        """
        pipes = (dj.U('pipe') & self).fetch('pipe')
        if len(pipes) != 1:
            raise PipelineException('Please restrict query to a single pipeline.')
        rel, _ = self.mapping[pipes[0]]
        return sys.modules[rel.__module__]

    def _make_tuples(self, key):
        # find the matching pipeline from those specified in self.mapping
        try:
            pipe, src, dest = next(((pipe, src, dest)
                for pipe, (src, dest) in self.mapping.items() if src() & key))
        except StopIteration:
            raise PipelineException('The key source yielded a key from an uknown pipeline')
        self.insert1({**key, 'pipe': pipe})
        dest().insert(src() & key, ignore_extra_fields=True)
        return src


@schema
class ScanSet(Resolver, dj.Computed):
    definition = """ # set of units in the same scan
    -> experiment.Scan
    -> shared.PipelineVersion
    -> shared.Field
    -> shared.Channel
    -> shared.SegmentationMethod
    ---
    -> Pipe
    """
    @property
    def key_source(self):
        return reso.ScanSet().proj() + meso.ScanSet().proj()

    @property
    def mapping(self):
        return {'reso': (reso.ScanSet, ScanSet.Reso),
                'meso': (meso.ScanSet, ScanSet.Meso)}

    class Unit(dj.Part):
        definition = """ # individual units corresponding to <module>.ScanSet.Unit
        -> experiment.Scan
        -> shared.PipelineVersion
        -> shared.SegmentationMethod
        unit_id                 : int       # unique per scan & segmentation method
        ---
        -> master                           # for it to act as a part table of ScanSet
        """

    class Reso(dj.Part):
        definition = """
        -> reso.ScanSet
        -> master
        """

    class Meso(dj.Part):
        definition = """
        -> meso.ScanSet
        -> master
        """

    def _make_tuples(self, key):
        src = super()._make_tuples(key)
        module = sys.modules[src.__module__]
        self.Unit().insert(self * module.ScanSet.Unit() & key, ignore_extra_fields=True)


@schema
class Activity(Resolver, dj.Computed):
    definition = """ # calcium activity for a single field within a scan
    -> experiment.Scan
    -> shared.PipelineVersion
    -> shared.Field
    -> shared.Channel
    -> shared.SegmentationMethod
    -> shared.SpikeMethod
    ---
    -> Pipe
    """
    @property
    def key_source(self):
        return reso.Activity().proj() + meso.Activity().proj()

    @property
    def mapping(self):
        return {'reso': (reso.Activity, Activity.Reso),
                'meso': (meso.Activity, Activity.Meso)}

    class Trace(dj.Part):
        definition = """
        # Trace corresponding to <module>.Activity.Trace
        -> master
        unit_id                 : int           # unique per scan & segmentation method
        """

    class Reso(dj.Part):
        definition = """
        -> reso.Activity
        -> master
        """

    class Meso(dj.Part):
        definition = """
        -> meso.Activity
        -> master
        """

    def _make_tuples(self, key):
        src = super()._make_tuples(key)
        module = sys.modules[src.__module__]
        self.Trace().insert(self * module.Activity.Trace() & key, ignore_extra_fields=True)


@schema
class ScanDone(Resolver, dj.Computed):
    definition = """ # calcium activity for the whole scan (multiple scan fields)
    -> experiment.Scan
    -> shared.PipelineVersion
    -> shared.SegmentationMethod
    -> shared.SpikeMethod
    ---
    -> Pipe
    """
    @property
    def key_source(self):
        return reso.ScanDone().proj() + meso.ScanDone().proj()

    @property
    def mapping(self):
        return {'reso': (reso.ScanDone, ScanDone.Reso),
                'meso': (meso.ScanDone, ScanDone.Meso)}

    class Reso(dj.Part):
        definition = """
        -> reso.ScanDone
        -> master
        """

    class Meso(dj.Part):
        definition = """
        -> meso.ScanDone
        -> master
        """


@schema
class MotionCorrection(Resolver, dj.Computed):
    definition = """ # calcium activity for the whole scan (multiple scan fields)
    -> experiment.Scan
    -> shared.PipelineVersion
    -> shared.Field
    ---
    -> Pipe
    """

    @property
    def key_source(self):
        return reso.MotionCorrection().proj() + meso.MotionCorrection().proj()

    @property
    def mapping(self):
        return {'reso': (reso.MotionCorrection, MotionCorrection.Reso),
                'meso': (meso.MotionCorrection, MotionCorrection.Meso)}

    class Reso(dj.Part):
        definition = """
        -> reso.MotionCorrection
        -> master
        """

    class Meso(dj.Part):
        definition = """
        -> meso.MotionCorrection
        -> master
        """
