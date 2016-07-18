import datajoint as dj
from . import experiment, preprocess

from distutils.version import StrictVersion

assert StrictVersion(dj.__version__) >= StrictVersion('0.2.7')

schema = dj.schema('pipeline_quality', locals())

