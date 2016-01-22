import datajoint as dj

schema = dj.schema('common_mice', locals())

@schema
class Death(dj.Manual):
    definition = None


@schema
class Founders(dj.Manual):
    definition = None


@schema
class Gels(dj.Manual):
    definition = None


@schema
class Genotypes(dj.Manual):
    definition = None


@schema
class Lanes(dj.Manual):
    definition = None


@schema
class Lines(dj.Manual):
    definition = None


@schema
class Mice(dj.Manual):
    definition = None


@schema
class Parents(dj.Manual):
    definition = None


@schema
class PcrPrimers(dj.Manual):
    definition = None


@schema
class PcrPrograms(dj.Manual):
    definition = None


@schema
class Primers(dj.Manual):
    definition = None

@schema
class Requests(dj.Manual):
    definition = None

@schema
class Transfers(dj.Manual):
    definition = None


