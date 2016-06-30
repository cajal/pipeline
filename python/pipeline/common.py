"""
This is a legacy schema that will no longer be used in the unified pipeline.
It is provided here to support the migration process.
"""

import datajoint as dj

from distutils.version import StrictVersion
assert StrictVersion(dj.__version__) >= StrictVersion('0.2.5')

schema = dj.schema('common', locals())

schema.spawn_missing_classes()
