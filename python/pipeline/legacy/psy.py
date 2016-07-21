import datajoint as dj
from . import common
from distutils.version import StrictVersion
assert StrictVersion(dj.__version__) >= StrictVersion('0.2.5')

schema = dj.schema('common_psy', locals())

schema.spawn_missing_classes()
