import datajoint as dj
schema = dj.schema('pipeline_monet', locals())
schema.spawn_missing_classes()


