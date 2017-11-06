import datajoint as dj
import pandas as pd
from . import mice  # needed for referencing
import numpy as np
from distutils.version import StrictVersion
import os
from commons import lab


schema = dj.schema('pipeline_experiment', locals(), create_tables=False)


def erd():
    """for convenience"""
    dj.ERD(schema).draw()


@schema
class Fluorophore(dj.Lookup):
    definition = """  # calcium-sensitive indicators

    fluorophore     : char(10)   # fluorophore short name
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
        ['OGB', ''],
        ['RCaMP1a', '']
    ]

    class EmissionSpectrum(dj.Part):
        definition = """  # spectra of fluorophores in Ca++ loaded and Ca++ free state

        -> Fluorophore
        loaded          : bool      # whether the spectrum is for Ca++ loaded or free state
        ---
        wavelength      : longblob  # wavelength in nm
        fluorescence    : longblob  # fluorescence in arbitrary units
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
    definition = """  # objective lens list
    lens    : char(4) # objective lens
    ---
    """
    contents = [['10x'], ['16x'], ['25x']]


@schema
class Rig(dj.Lookup):
    definition = """
    rig     : char(8)    # multiphoton imaging setup
    ---
    """
    contents = [
        ['2P1'],  # aod scans
        ['2P2'],  # resonant scans (Shan)
        ['2P3'],  # resonant scans
        ['2P4'],  # mesoscope scans
        ['2P5'],  # resonant
        ['3P1'],  # 3-photon galvo scans
        ['2P3P1']  # 2-photon resonant, 3-photon galvo
    ]


@schema
class FOV(dj.Lookup):
    definition = """  # field-of-view sizes for all lenses and magnifications
    -> Rig
    -> Lens
    mag         : decimal(5,2)  # ScanImage zoom factor
    fov_ts      : datetime      # fov measurement date and time
    ---
    height      : decimal(5,1)  # measured height of field of view along axis of pipette (medial/lateral on mouse)
    width       : decimal(5,1)  # measured width of field of view perpendicular to pipette (rostral/caudal on mouse)
    """


@schema
class Anesthesia(dj.Lookup):
    definition = """   #  anesthesia states
    anesthesia                     : char(20) # anesthesia short name
    ---
    anesthesia_description=''       : varchar(255) # longer description
    """
    contents = [
        ['awake', ''],
        ['fentanyl', ''],
        ['isoflurane', '']
    ]


@schema
class Person(dj.Lookup):
    definition = """  # person information

    username      : char(12)   # lab member
    ---
    full_name     : varchar(255)
    """
    contents = [
        ['unknown', 'placeholder'],
        ['atlab', 'general atlab account'],
        ['cathryn', 'Cathryn Rene Cadwell'],
        ['dimitri', 'Dimitri Yatsenko'],
        ['ecobost', 'Erick Cobos T'],
        ['edgar', 'Edgar Y. Walker'],
        ['fabee', 'Fabian Sinz'],
        ['jake', 'Jacob Reimer'],
        ['jiakun', 'Jiakun Fu'],
        ['manolis', 'Emmanouil Froudarakis'],
        ['minggui', 'Minggui Chen'],
        ['Paul', 'hey paul'],
        ['shan', 'Shan Shen'],
        ['shuang', 'Shuang Li'],
        ['xiaolong', 'Xiaolong Jiang']
    ]


@schema
class BrainArea(dj.Lookup):
    definition = """

    brain_area          : char(12)     # short name for cortical area
    ---
    area_description    : varchar(255)
    """
    contents = [
        ['unset', ''],
        ['unknown', ''],
        ['V1', ''],
        ['LM', ''],
        ['LI', ''],
        ['A', ''],
        ['AL', ''],
        ['AM', ''],
        ['P', ''],
        ['PM', ''],
        ['POR', ''],
        ['RL', ''],
        ['other', ''],
    ]


@schema
class Layer(dj.Lookup):
    definition = """

    layer                : char(12)     # short name for cortical layer
    ---
    layer_description    : varchar(255)
    z_start=null         : float        # starting depth
    z_end=null           : float        # deepest point
    """
    contents = [
        ['L1', '', 0, 100],
        ['L2/3', '', 100, 370],
        ['L4', '', 370, 500],
        {'layer': 'unset', 'layer_description': ''}
    ]

    def get_layers(self, z):
        l, fr, to = self.fetch('layer', 'z_start', 'z_end')
        m = np.vstack([(z > f) & (z < t) for f,t in zip(fr, to)]).T
        return np.hstack([l[mm] for mm in m]).squeeze()



@schema
class Software(dj.Lookup):
    definition = """ # recording software information

    software        : varchar(20) # name of the software
    version         : char(10)    # version
    ---
    """
    contents = [
        ['unset', '0.0'],
        ['aod', '2.0'],
        ['imager', '1.0'],
        ['scanimage', '4.0'],
        ['scanimage', '4.2'],
        ['scanimage', '4.2pr1'],
        ['scanimage', '5.1'],
        ['scanimage', '5.2'],
        ['scanimage', '2016b'],
        ['scanimage', '2016b3P']
    ]


@schema
class Compartment(dj.Lookup):
    definition = """  # cell compartments that can be imaged

    compartment         : char(16)
    ---
    """
    contents = [['axon'], ['soma'], ['bouton']]


@schema
class PMTFilterSet(dj.Lookup):
    definition = """  # microscope filter sets: dichroic and PMT Filters

    pmt_filter_set          : varchar(16)       # short name of microscope filter set
    ----
    primary_dichroic        :  varchar(255)     #  passes the laser  (excitation/emission separation)
    secondary_dichroic      :  varchar(255)     #  splits emission spectrum
    filter_set_description  :  varchar(4096)    #  A detailed description of the filter set
    """
    contents = [
        ['2P3 red-green A', '680 nm long-pass?', '562 nm long-pass', 'purchased with Thorlabs microscope'],
        ['2P3 blue-green A', '680 nm long-pass?', '506 nm long-pass', 'purchased with Thorlabs microscope']]

    class Channel(dj.Part):
        definition = """  # PMT description including dichroic and filter

        -> PMTFilterSet
        pmt_channel : tinyint   #  pmt_channel
        ---
        color      : enum('green', 'red', 'blue')
        pmt_serial_number   :  varchar(40)   #
        spectrum_center     :  smallint  unsigned  #  (nm) overall pass spectrum of all upstream filters
        spectrum_bandwidth  :  smallint  unsigned  #  (nm) overall pass spectrum of all upstream filters
        pmt_filter_details  :  varchar(255)  #  more details, spectrum, pre-amp gain, pre-amp ADC filter
        """
        contents = [
            ['2P3 red-green A', 1, 'green', 'AC7438 Thor', 525, 50, ''],
            ['2P3 red-green A', 2, 'red', 'AC7753 Thor', 625, 90, ''],
            ['2P3 blue-green A', 1, 'blue', 'AC7438 Thor', 475, 50, ''],
            ['2P3 blue-green A', 2, 'green', 'AC7753 Thor', 540, 50, '']
        ]


@schema
class LaserCalibration(dj.Manual):
    definition = """  # stores measured values from the laser power calibration

    -> Rig
    calibration_ts      : timestamp         # calibration timestamp -- automatic
    ---
    -> Lens
    -> Person
    -> Software
    """

    class PowerMeasurement(dj.Part):
        definition = """
       -> LaserCalibration
        wavelength      : int                 # wavelength of the laser
        attenuation     : decimal(4,1)        # power setting (percent for resonant scanner or degrees of polarizer)
        bidirectional   : bool                # 0 if off 1 if on
        gdd             : int                 # GDD setting on the laser
        ---
        power           : float               # power in mW
        """

    def plot_calibration_curve(self, calibration_date, rig):
        import matplotlib.pyplot as plt
        import seaborn as sns
        session = LaserCalibration.PowerMeasurement() & dict(calibration_date=calibration_date, rig=rig)
        sns.set_context('talk')
        with sns.axes_style('darkgrid'):
            fig, ax = plt.subplots()

        # sns.set_palette("husl")

        for k in (dj.U('pockels', 'bidirectional', 'gdd', 'wavelength') & session).fetch.keys():
            pe, po, zoom = (session & k).fetch('percentage', 'power', 'zoom')
            zoom = np.unique(zoom)
            ax.plot(pe, po, 'o-', label=(u"zoom={0:.2f} ".format(zoom[0])
                                         + " ".join("{0}={1}".format(*v) for v in k.items())))
        ax.legend(loc='best')
        ax.set_xlim((0, 100))
        y_min, y_max = [np.round(y / 5) * 5 for y in ax.get_ylim()]
        ax.set_yticks(np.arange(0, y_max + 5, 5))
        ax.set_xlabel('power [in %]')
        ax.set_ylabel('power [in mW]')

        return fig, ax


@schema
class Session(dj.Manual):
    definition = """  # imaging session

    -> mice.Mice
    session                       : smallint            # session index for the mouse
    ---
    -> Rig
    session_date                  : date                # date
    -> Person
    -> Anesthesia
    scan_path                     : varchar(255)        # file path for TIFF stacks
    behavior_path =""             : varchar(255)        # pupil movies, whisking, locomotion, etc.
    craniotomy_notes=""           : varchar(4095)       # free-text notes
    session_notes=""              : varchar(4095)       # free-text notes
    session_ts=CURRENT_TIMESTAMP  : timestamp           # automatic
    """

    class Fluorophore(dj.Part):
        definition = """  # fluorophores expressed in prep for the imaging session

        -> Session
        -> Fluorophore
        ---
        notes=""        : varchar(255)  # additional information about fluorophore in this scan
        """

    class TargetStructure(dj.Part):
        definition = """  # specifies which neuronal structure was imaged

        -> Session
        -> Fluorophore
        -> Compartment
        ---
        """

    class PMTFilterSet(dj.Part):
        definition = """

        -> Session
        ---
        -> PMTFilterSet
        """


@schema
class Aim(dj.Lookup):
    definition = """  # declared purpose of the scan

    aim                  : varchar(36)                  # short name for the purpose of the scan
    ---
    aim_description      : varchar(255)
    """


class HasFilename:
    """ Mixin to add local_filenames_as_wildcard property to Scan and Stack. """
    @property
    def local_filenames_as_wildcard(self):
        """Returns the local filename for all parts of this scan (ends in *.tif)."""
        scan_path = (Session() & self).fetch1('scan_path')
        local_path = lab.Paths().get_local_path(scan_path)

        scan_name = (self.__class__() & self).fetch1('filename')
        local_filename = os.path.join(local_path, scan_name) + '*.tif'  # all parts

        return local_filename


@schema
class Scan(dj.Manual, HasFilename):
    definition = """    # scanimage scan info

    -> Session
    scan_idx                : smallint              # number of TIFF stack file
    ---
    -> Lens
    -> BrainArea
    -> Aim
    filename                : varchar(255)          # file base name
    depth=0                 : int                   # (um) manual depth measurement with respect to the surface of the cortex where fastZ = 0
    scan_notes              : varchar(4095)         # free-notes
    site_number=0           : tinyint               # site number
    -> Software
    scan_ts                 : timestamp             # don't edit
    """

    class EyeVideo(dj.Part):
        definition = """  # name of the eye tracking video

        -> Scan
        ---
        filename            : varchar(50)                   # filename of the video
        """

    class BehaviorFile(dj.Part):
        definition = """  # name of the running wheel file

        -> Scan
        ---
        filename            : varchar(50)                   # filename of the video
        """

    class Laser(dj.Part):
        definition = """  # laser parameters for the scan

        -> Scan
        ---
        wavelength          : float                         # (nm)
        power               : float                         # (mW) to brain
        gdd                 : float                         # gdd setting
        """

@schema
class Stack(dj.Manual):
    definition = """ # structural stack information
    -> Session
    stack_idx               : smallint              # id of the stack
    ---
    -> Lens
    -> BrainArea
    -> Aim
    -> Software
    top_depth               : smallint              # (um) depth at top of the stack
    bottom_depth            : smallint              # (um) depth at bottom of stack
    stack_notes             : varchar(4095)         # free notes
    stack_ts=CURRENT_TIMESTAMP : timestamp          # don't edit
    """

    class Filename(dj.Part, HasFilename):
        definition = """ # filenames that compose one stack (used in resonant and 3p scans)

        -> Stack
        filename_idx        : tinyint               # id of the file
        ---
        filename            : varchar(255)          # file base name
        surf_depth=0        : float                 # ScanImage's z at cortex surface
        """

    class Laser(dj.Part):
        definition = """  # laser parameters for the stack

        -> Stack
        ---
        wavelength          : int                   # (nm)
        max_power           : float                 # (mW) to brain
        gdd                 : float                 # gdd setting
        """

@schema
class ScanIgnored(dj.Manual):
    definition = """  # scans to ignore
    -> Scan
    """

schema.spawn_missing_classes()
