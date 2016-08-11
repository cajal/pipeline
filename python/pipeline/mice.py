"""
The `common_mice` schema is maintained by another package and is included here for ease of reference.
DO NOT create new tables here.
"""

import datajoint as dj
schema = dj.schema('common_mice', locals())
schema.spawn_missing_classes()
