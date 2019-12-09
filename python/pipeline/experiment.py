import datajoint as dj
import pandas as pd
from . import mice  # needed for referencing
import numpy as np
import os
from commons import lab


schema = dj.schema('pipeline_experiment', locals(), create_tables=False)


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
        ['xiaolong', 'Xiaolong Jiang'],
        ['taliah', 'Taliah Muhammad'],
        ['zhiwei',  'Zhiwei Ding']
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
class MonitorCalibration(dj.Manual):
    definition = """ # monitor luminance calibration
    
    -> experiment.Scan
    ---
    pixel_value             : mediumblob      # control pixel value (0-255)
    luminance               : mediumblob      # luminance in cd/m^2
    amplitude               : float           # lum = Amp*pixel_value^gamma + offset
    gamma                   : float           #
    offset                  : float           #
    mse                     : float           #
    ts                      : timestamp       # timestamp
    """

    def plot_calibration_curve(self):
        import matplotlib.pyplot as plt

        # Get data
        pixel_values, luminances = self.fetch1('pixel_value', 'luminance')

        # Plot original data
        fig = plt.figure()
        plt.plot(pixel_values, luminances, label='Data')
        plt.xlabel('Pixel intensities')
        plt.ylabel('Luminance (cd/m^2)')

        # Plot fit
        amp, gamma, offset = self.fetch1('amplitude', 'gamma', 'offset')
        xs = np.arange(255) # pixel
        ys = amp * (xs ** gamma) + offset
        plt.plot(xs, ys, label='Fit')

        plt.legend()

        return fig


@schema
class MouseRoom(dj.Lookup):
    definition = """ # Mouse location after surgery
    mouse_room                : varchar(64)         # Building letter along with room number
    """
    contents = [
        ['T019'],
        ['T057'],
        ['T082C'],
        ['Other'],
    ]


@schema
class SurgeryType(dj.Lookup):
    definition = """ # diff types of surgery
    
    surgery_type                : varchar(64)         # Types of surgery performed on mice
    """
    contents = [
        ['Cranial Window and Headbar'],
        ['Headbar'],
        ['Burr Hole and Suture'],
        ['C-Section'],
        ['Viral Injection'],
        ['Electroporation']
    ]


@schema
class SurgeryOutcome(dj.Lookup):
    definition = """ # surgery outcomes
    
    surgery_outcome             : varchar(32)         # Possible outcomes of surgeries performed on mice
    """
    contents = [
        ['Survival'],
        ['Non-Survival'],
    ]


@schema
class Surgery(dj.Manual):
    definition = """ # surgeries performed on mice
    -> mice.Mice
    surgery_id                   : smallint               # Unique number given to each surgery
    ---
    date                         : date                   # YYYY-MM-DD Format. Date surgery was performed
    time                         : time                   # Start of mouse recovery
    -> Person
    -> MouseRoom            
    -> SurgeryOutcome
    -> SurgeryType
    surgery_quality              : tinyint                # 0-5 self-rating, 0 being worst and 5 best
    ketoprofen = null            : decimal(4,3) unsigned  # Amount of Ketoprofen given to mouse
    weight = null                : decimal(5,2) unsigned  # Weight of mouse before surgery
    surgery_notes = ""           : varchar(256)           # Notes on surgery
    """


@schema
class SurgeryStatus(dj.Manual):
    definition = """ # updates to surgeries
    
    -> Surgery
    timestamp                           : timestamp              # Timestamp of entry
    ---
    euthanized = 0                      : boolean
    day_one = 0                         : boolean                # First day checkup performed
    day_two = 0                         : boolean                # Second day checkup performed
    day_three = 0                       : boolean                # Third day checkup performed
    checkup_notes = ""                  : varchar(265)           # Notes on surgery checkups
    """


@schema
class Session(dj.Manual):
    definition = """  # imaging session

    -> mice.Mice
    session                      : smallint            # session index for the mouse
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

        scan_name = (self.__class__() & self.proj()).fetch1('filename')
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
    depth=0                 : int                   # (um) manual depth measurement of the cortex surface
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
    surf_depth=0            : smallint              # (um) depth of the surface of the cortex
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


@schema
class Fixes(dj.Manual):
    definition = """ # any fixes or known problems for a scan
    -> Scan
    """

    class IrregularTimestamps(dj.Part):
        definition = """ # produced by packets dropped or not read fast enough (slow computer)

        -> master
        ----
        num_gaps      :int          # number of gaps of bad data in the signal
        num_secs      :float        # number of seconds of bad data in the signal
        """


@schema
class AutoProcessing(dj.Manual):
    definition = """ # scans that should enter automatic processing
    -> Scan
    ---
    priority=0          :tinyint       # highest priority is processed first
    autosegment=false   :boolean       # segment somas in the first channel with default method
    """

@schema
class ProjectorColor(dj.Lookup):
    definition = """
    # color options for projector channels
    color               : varchar(32)               # color name
    ---
    """
    contents = [
        ['none'],
        ['red'],
        ['green'],
        ['blue'],
        ['UV']
    ]

@schema
class ProjectorConfig(dj.Lookup):
    definition = """
    # projector configuration
    projector_config_id         : tinyint                   # projector config    
    ---
    -> ProjectorColor.proj(channel_1="color")               # channel 1 means 1st color channel. Usually red
    -> ProjectorColor.proj(channel_2="color")               # channel 2 means 2nd color channel. Usually green
    -> ProjectorColor.proj(channel_3="color")               # channel 3 means 3rd color channel. Usually blue
    refresh_rate                : float                     # refresh rate in Hz

    """
    contents = [
        [0, 4, 2, 3, 60],
        [1, 4, 4, 2, 60],
        [2, 4, 4, 2, 30]
    ]


@schema
class Projector(dj.Lookup):
    definition = """
    # projector specifications
    projector_id        : tinyint                               # projector id
    ---
    pixel_width         : smallint                              # number of pixels in width
    pixel_height        : smallint                              # number of pixels in height
    """
    contents = [
        [0, 1920, 1080],
        [1, 1140, 912]
    ]


@schema
class ProjectorSetup(dj.Lookup):
    definition = """
    # projector set up
    -> Projector
    -> ProjectorConfig
    ---
    display_width       : float         # projected display width in cm
    display_height      : float         # projected display height in cm
    target_distance     : float         # distance from mouse to the display in cm
    """


schema.spawn_missing_classes()
