import datajoint as dj

schema = dj.schema('pipeline_em', locals())


@schema
class Stack(dj.Manual):
    definition = """
    # EM Stack

    em              : smallint unsigned    # Seung Lab stack identifier
    ---
    stack_boss_id   : char(64)             # ID in the BOSS
    """


# TODO:
# Will the neuron ID in the BOSS be the same as this neuron ID? If not, we should add the BOSS ID
@schema
class Neuron(dj.Manual):
    definition = """
    #  anatomically identified neurons

    -> Stack
    neuron_id   : int  unsigned           # Seung Lab neuron identifier
    ---
    """


@schema
class FragmentType(dj.Lookup):
    definition = """
    # Lookup table for fragment types

    fragment_type      : char(10) # fragment type as name
    ---
    """

    contents = list(zip(['dendrite', 'axon', 'soma', 'spine']))


# TODO:
# Will the fragment ID in the BOSS be the same as this fragment ID? If not, we should add the BOSS ID
@schema
class Fragment(dj.Manual):
    definition = """
    -> Neuron
    fragment_id     : int unsigned
    ---
    -> FragmentType
    """


@schema
class KeyPoint(dj.Manual):
    definition = """ # appproximate centroid of the neuron's centroid
    -> Neuron
    ---
    x : double  # (um) -- EM scan coordinate system
    y : double  # (um)
    z : double  # (um)
    """


# TODO:
# * Does a volume always belong to an object/fragment? If so, it should be a child of it.
# * If every neuron/fragment has a bounding box, we should put the attributes there.
@schema
class BoundingBox(dj.Manual):
    definition = """
    # Bounding box of fragments -- what can have a bounding box?
    ->
    box_id  : int
    ----
    x1      : float # (um)
    y1      : float # (um)
    z1      : float # (um)
    x2      : float # (um)
    y2      : float # (um)
    z2      : float # (um)
    """


# TODO:
# Does a volume always belong to an object/fragment? If so, it should be a child of it.
@schema
class Volume(dj.Manual):
    definition = """
    # set of voxels that belong to an object

    volume_id :  int
    ---
    vol_boss_id :  char(64)  # volume ID in the BOSS
    """


# TODO:
# Should there be two types of synapses, i.e. chemical vs. electrical?
@schema
class Synapse(dj.Manual):
    definition = """
    # synapse between two neurons

    synapse_id          : int unsigned # unique identifier of
    ---
    presynaptic -> Fragment
    postsynaptic -> Fragment
    """
