from distutils.version import StrictVersion
import sys
import datajoint as dj
from . import experiment, reso, meso, shared
from .exceptions import PipelineException

assert StrictVersion(dj.__version__) >= StrictVersion('0.8.2'), "Please upgrade datajoint to 0.8.2+"


schema = dj.schema('pipeline_fuse', locals())


@schema
class ScanDone(dj.Computed):
    definition = """
    # Calcium Activity from multiple pipelines
    -> experiment.Scan
    -> shared.PipelineVersion
    -> shared.SpikeMethod
    -> shared.SegmentationMethod
    ---
    pipe : char(8)   # meso, reso, etc
    """

    key_source = meso.ScanDone().proj() + reso.ScanDone().proj()

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

    def _make_tuples(self, key):

        for pipe, src, dest in (
                ('meso', meso.ScanDone, ScanDone.Meso),
                ('reso', reso.ScanDone, ScanDone.Reso)):
            if src() & key:
                break
        else:
            raise PipelineException(
                    'The key source yielded a key from a new pipeline')

        self.insert1(dict(key, pipe=pipe))
        dest().insert(src() & key, ignore_extra_fields=True)


    def resolve(self):
        """
        Given a fuse.ScanDone() object, return the corresponding
        module.ScanDone() object and the module itself.
        :return: scanDone,  module
        """
        if len(dj.U('pipe') & self) != 1:
            raise PipelineException('cannot query from multiple pipelines at once.  Please narrow down your restriction')
        scanDone = (reso.ScanDone() & self) or (meso.ScanDone() & self)
        return scanDone, sys.modules[scanDone.__module__]

