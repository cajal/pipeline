import datajoint as dj
from . import mice

from distutils.version import StrictVersion
assert StrictVersion(dj.__version__) >= StrictVersion('0.2.5')

schema = dj.schema('pipeline_experiment', locals())


@schema
class Dye(dj.Lookup):
    definition = """
    # calcium-sensitive indicators
    dye                   : char(10)   # fluorophore short name
    -----
    dye_description = ''  : varchar(2048)
    """


@schema
class Lens(dj.Lookup):
    definition = """
    # objective lens list
    lens            : char(4)                # objective lens
    ---
    """


@schema
class FOV(dj.Lookup):
    definition = """
    # field-of-view sizes for all lenses and magnifications
    setup           : char(4)                # two-photon setup
    -> Lens
    mag                         : decimal(5,2)                  # ScanImage zoom factor
    fov_ts                      : datetime                      # fov measurement date and time
    ---
    height                      : decimal(5,1)                  # measured width of field of view along axis of pipette (medial/lateral on mouse)
    width                       : decimal(5,1)                  # measured width of field of view perpendicular to pipette (rostral/caudal on mouse)
    """


@schema
class Anesthesia(dj.Lookup):
    definition = """
    # different anesthesia

    anesthesia                     : char(20) # anesthesia short name
    ---
    anesthesia_description=''      : varchar(255) # longer description
    """

    contents = [
        ('isoflurane', ''),
        ('fentanyl', ''),
        ('awake', '')
    ]


@schema
class Person(dj.Lookup):
    definition = """
    # person information
    username      : char(12)
    ---
    full_name     : varchar(255)
    """
    contents = [
        ('manolis', 'Emmanouil Froudarakis'),
        ('dimitri', 'Dimitri Yatsenko'),
        ('shan', 'Shan Shen'),
        ('jake', 'Jacob Reimer (Overlord)'),
        ('fabee', 'Fabian Sinz'),
        ('edgar', 'Edgar Y. Walker'),
        ('cathryn', 'Cathryn Rene Cadwell'),
        ('shuang', 'Shuang Li'),
        ('xiaolong', 'Xiaolong Jiang (Patchgrandmaster)'),
    ]


@schema
class Session(dj.Manual):
    definition = """
    # session

    -> mice.Mice
    session                       : smallint                      # session index
    ---
    -> Anesthesia
    -> Person
    session_date                  : date                          # date
    scan_path                     : varchar(255)                  # file path for TIFF stacks
    craniotomy_notes              : varchar(4095)                 # free-text notes
    session_notes                 : varchar(4095)                 # free-text notes
    session_ts=CURRENT_TIMESTAMP  : timestamp                     # automatic
    """


@schema
class BrainArea(dj.Lookup):
    definition = """
    cortical_area       : char(12)     # short name for cortical area
    ---
    area_description    : varchar(255)
    """
    contents = [
        ('other', ''),
        ('unknown', ''),
        ('V1', ''),
        ('LM', ''),
        ('AL', ''),
        ('PM', ''),
     ]


@schema
class Software(dj.Lookup):
    definition = """
    # recording software information
    software        : varchar(20) # name of the software
    version         : char(10)    # version
    ---
    """
    contents = [
        ('scanimage', '3.8'),
        ('scanimage', '4.0'),
        ('aod', '2.0'),
        ('imager', '1.0')]


@schema
class Scan(dj.Manual):
    definition = """
    # scanimage scan info
    -> Session
    scan_idx        : smallint               # number of TIFF stack file
    ---
    -> Lens
    -> BrainArea
    laser_wavelength            : float                         # (nm)
    laser_power                 : float                         # (mW) to brain
    filename                    : varchar(255)                  # file base name
    scan_notes                  : varchar(4095)                 # free-notes
    structural=0                : boolean                       # was the scan structural or not
    surf_z=0                    : int                           # manual depth measurement
    site_number=0               : tinyint                       # site number
    -> Software
    scan_ts=CURRENT_TIMESTAMP   : timestamp                     # don't edit
    """


@schema
class SessionDye(dj.Manual):
    definition = """
    # Dye used in session
    -> Session
    -> Dye
    ---
    notes           : varchar(255) # additional information
    """


@schema
class PMTFilter(dj.Lookup):
    definition = """
    # filter in the filter cube
    pmt_filter :char(12)   # PMT filter_description
    ---
    pmt_filter_details :varchar(255)  #  more details, spectrum
    """


@schema
class PMTChannel(dj.Manual):
    definition = """
    # microscope acquisition channel
    -> Session
    pmt_channel : tinyint  #  two-photon channel
    ---
    -> PMTFilter
    notes           : varchar(255) # additional information
    """


schema.spawn_missing_classes()