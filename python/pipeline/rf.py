import datajoint as dj
schema = dj.schema('pipeline_rf', locals())


@schema
class Session(dj.Manual):
    definition = ...


@schema
class Site(dj.Manual):
    definition = ...


@schema
class Scan(dj.Manual):
    definition = ...


@schema
class VolumeSlice(dj.Lookup):
    definition =  ...


@schema
class Eye(dj.Imported):
    definition = ...

    def _make_tuples(self, key):
        raise dj.DataJointError("Eye is populated from matlab! What a shame!")

@schema
class EyeFrame(dj.Manual):
    definition = ...