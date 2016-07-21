import datajoint as dj
import pandas as pd
from . import mice
import numpy as np
from distutils.version import StrictVersion
import numpy as np
import inspect
import os

assert StrictVersion(dj.__version__) >= StrictVersion('0.2.7')

schema = dj.schema('pipeline_experiment', locals())


def erd():
    """for convenience"""
    dj.ERD(schema).draw()


@schema
class Fluorophore(dj.Lookup):
    definition = """
    # calcium-sensitive indicators
    fluorophore          : char(10)   # fluorophore short name
    -----
    dye_description = ''  : varchar(2048)
    """
    contents = [
        ['GCaMP6s', ''],
        ['GCaMP6f', ''],
        ['Twitch2B', ''],
        ['mRuby', ''],
        ['mCherry', ''],
        ['tdTomato', ''],
        ['OGB', '']]

    class EmissionSpectrum(dj.Part):
        definition = """
        # spectra of fluorophores in Ca++ loaded and Ca++ free state
        ->Fluorophore
        loaded          : bool # whether the spectrum is for Ca++ loaded or free state
        ---
        wavelength      : longblob # wavelength in nm
        fluorescence    : longblob # fluorescence in arbitrary units
        """

        @property
        def contents(self):
            # yield Twitch2B spectra
            if len(self & dict(fluorophore='Twitch2B')) < 2:
                path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
                loaded = pd.read_csv(path + '/data/twitch2B_loaded.csv')
                free = pd.read_csv(path + '/data/twitch2B_free.csv')
                x = np.linspace(np.min(free.wavelength), np.max(free.wavelength), 100)
                y_loaded = np.interp(x, loaded.wavelength, loaded.fluorescence)
                y_free = np.interp(x, free.wavelength, free.fluorescence)
                yield ('Twitch2B', True, x, y_loaded)
                yield ('Twitch2B', False, x, y_free)


@schema
class Lens(dj.Lookup):
    definition = """
    # objective lens list
    lens            : char(4)                # objective lens
    ---
    """

    contents = [['16x'], ['25x']]


@schema
class Rig(dj.Lookup):
    definition = """
    rig : char(4)    # multiphoton imaging setup
    ---
    """
    contents = [['2P2'], ['2P3']]


@schema
class FOV(dj.Lookup):
    definition = """  # field-of-view sizes for all lenses and magnifications
    -> Rig
    -> Lens
    mag                         : decimal(5,2)                  # ScanImage zoom factor
    fov_ts                      : datetime                      # fov measurement date and time
    ---
    height                      : decimal(5,1)                  # measured width of field of view along axis of pipette (medial/lateral on mouse)
    width                       : decimal(5,1)                  # measured width of field of view perpendicular to pipette (rostral/caudal on mouse)
    """


@schema
class Anesthesia(dj.Lookup):
    definition = """   #  anesthesia states
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
    username      : char(12)   # lab member
    ---
    full_name     : varchar(255)
    """
    contents = [
        ('unknown', 'placeholder'),
        ('jiakun', 'Jiakun Fu'),
        ('manolis', 'Emmanouil Froudarakis'),
        ('dimitri', 'Dimitri Yatsenko'),
        ('shan', 'Shan Shen'),
        ('jake', 'Jacob Reimer'),
        ('fabee', 'Fabian Sinz'),
        ('edgar', 'Edgar Y. Walker'),
        ('cathryn', 'Cathryn Rene Cadwell'),
        ('shuang', 'Shuang Li'),
        ('xiaolong', 'Xiaolong Jiang')
    ]


@schema
class BrainArea(dj.Lookup):
    definition = """
    brain_area       : char(12)     # short name for cortical area
    ---
    area_description    : varchar(255)
    """
    contents = [
        ('other', ''),
        ('unset', ''),
        ('unknown', ''),
        ('V1', ''),
        ('LM', ''),
        ('LI', ''),
        ('AL', ''),
        ('PM', ''),
        ('POR', ''),
        ('RL', '')
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
        ('unset', '0.0'),
        ('scanimage', '3.8'),
        ('scanimage', '4.0'),
        ('scanimage', '4.2'),
        ('scanimage', '4.2pr1'),
        ('scanimage', '5.1'),
        ('scanimage', '5.2'),
        ('aod', '2.0'),
        ('imager', '1.0')]


@schema
class Aim(dj.Lookup):
    definition = """  # what is being imaged (e.g. somas, axon) and why
    aim : varchar(40)   # short description of what is imaged and why
    """

    contents = [['unset'],
                ['functional: somas'],
                ['functional: axons'],
                ['functional: axons, somas'],
                ['functional: axons-green, somas-red'],
                ['functional: axons-red, somas-green'],
                ['structural']]


@schema
class PMTFilterSet(dj.Lookup):
    definition = """  #  microscope filter sets: dichroic and PMT Filters
    pmt_filter_set : varchar(16)    # short name of microscope filter set
    ----
    primary_dichroic      :  varchar(255)    #  passes the laser  (excitation/emission separation)
    secondary_dichroic    :  varchar(255)    #  splits emission spectrum
    filter_set_description :  varchar(4096)     #   A detailed description of the filter set
    """
    contents = [
        ['2P3 red-green A', '680 nm long-pass?', '562 nm long-pass', 'purchased with Thorlabs microscope'],
        ['2P3 blue-green A', '680 nm long-pass?', '506 nm long-pass', 'purchased with Thorlabs microscope']]

    class Channel(dj.Part):
        definition = """  #  PMT description including dichroic and filter
        -> PMTFilterSet
        pmt_channel : tinyint   #  pmt_channel
        ---
        color      : enum('green', 'red', 'blue')
        pmt_serial_number :  varchar(40)   #
        spectrum_center     :  smallint  unsigned  #  (nm) overall pass spectrum of all upstream filters
        spectrum_bandwidth  :  smallint  unsigned  #  (nm) overall pass spectrum of all upstream filters
        pmt_filter_details :varchar(255)  #  more details, spectrum, pre-amp gain, pre-amp ADC filter
        """
        contents = [
            ['2P3 red-green A', 1, 'green', 'AC7438 Thor', 525, 50, ''],
            ['2P3 red-green A', 2, 'red', 'AC7753 Thor', 625, 90, ''],
            ['2P3 blue-green A', 1, 'blue', 'AC7438 Thor', 475, 50, ''],
            ['2P3 blue-green A', 2, 'green', 'AC7753 Thor', 540, 50, '']
        ]


@schema
class Session(dj.Manual):
    definition = """ # imaging session
    -> mice.Mice
    session                       : smallint                      # session index for the mouse
    ---
    -> Rig
    session_date                  : date                          # date
    -> Person
    -> Anesthesia
    -> PMTFilterSet
    scan_path                     : varchar(255)                  # file path for TIFF stacks
    behavior_path =""             : varchar(255)   # pupil movies, whisking, locomotion, etc.
    craniotomy_notes=""           : varchar(4095)                 # free-text notes
    session_notes=""              : varchar(4095)                 # free-text notes
    session_ts=CURRENT_TIMESTAMP  : timestamp                     # automatic
    """

    class Fluorophore(dj.Part):
        definition = """
        # Fluorophores expressed in prep for the imaging session
        -> Session
        -> Fluorophore
        ---
        notes=""          : varchar(255) # additional information about fluorophore in this scan
        """


@schema
class Scan(dj.Manual):
    definition = """    # scanimage scan info
    -> Session
    scan_idx        : smallint               # number of TIFF stack file
    ---
    -> Lens
    -> BrainArea
    laser_wavelength            : float                         # (nm)
    laser_power                 : float                         # (mW) to brain
    filename                    : varchar(255)                  # file base name
    behavior_filename=""        : varchar(255)   # pupil movies, whisking, locomotion, etc.
    -> Aim
    depth=0                     : int                           # manual depth measurement
    scan_notes                  : varchar(4095)                 # free-notes
    site_number=0               : tinyint                       # site number
    -> Software
    scan_ts=CURRENT_TIMESTAMP   : timestamp                     # don't edit
    """


@schema
class ScanIgnored(dj.Manual):
    definition = """  # scans to ignore
    -> Scan
    """


schema.spawn_missing_classes()


def migrate_reso_pipeline():
    """
    migration from the old schema
    :return:
    """
    from . import common, rf, psy
    # migrate FOV calibration
    FOV().insert(rf.FOV().proj('width', 'height', rig="setup", fov_ts="fov_date").fetch(), skip_duplicates=True)

    # migrate Session
    sessions_to_migrate = rf.Session() * common.Animal() & 'session_date>"2016-02"' & 'animal_id>0'
    w = sessions_to_migrate.proj(
        'session_date',
        'anesthesia',
        'session_ts',
        'scan_path',
        rig='setup',
        username='lcase(owner)',
        pmt_filter_set='"2P3 red-green A"',
        session_notes="concat(session_notes,';;', animal_notes)")
    Session().insert(w.fetch(), skip_duplicates=True)

    # migrate fluorophore
    Session.Fluorophore().insert(sessions_to_migrate.proj('fluorophore').fetch(), skip_duplicates=True)

    assert len(Session()) == len(Session.Fluorophore())

    # migrate scans

    scans = (rf.Session().proj('lens', 'file_base') * rf.Scan()).proj(
        'lens',
        'laser_wavelength',
        'laser_power',
        'scan_notes',
        'scan_ts',
        'depth',
        software="'scanimage'",
        version="5.1",
        site_number='site',
        filename="concat(file_base, '_', LPAD(file_num, 5, '0'))",
        brain_area='cortical_area',
        aim="'unset'"
    ) & Session()

    Scan().insert(scans.fetch(as_dict=True), skip_duplicates=True)
