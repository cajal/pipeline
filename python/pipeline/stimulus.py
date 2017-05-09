import datajoint as dj

schema = dj.schema('pipeline_stimulus', locals())

schema.spawn_missing_classes()