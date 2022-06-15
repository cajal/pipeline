import numpy as np
import datajoint as dj
from getpass import getpass

class DJServer:
    """
    DataJoint connection class which stores a connection to the specified
    database and automatically adds expected schemas as attributes.

        Args:
            host (str)    : String containing the URL of the database to
                            connect to
            user (str)    : String of username for specified database
            password (str): String of password for specified database

    """

    def __init__(self, host, user, password):
        self.conn = dj.Connection(host=host, user=user, password=password)
        self.stimulus = dj.create_virtual_module(
            "stimulus", "pipeline_stimulus", connection=self.conn
        )
        self.meso = dj.create_virtual_module(
            "meso", "pipeline_meso", connection=self.conn
        )
        self.reso = dj.create_virtual_module(
            "reso", "pipeline_reso", connection=self.conn
        )
        self.pupil = dj.create_virtual_module(
            "pupil", "pipeline_eye", connection=self.conn
        )
        self.treadmill = dj.create_virtual_module(
            "treadmill", "pipeline_treadmill", connection=self.conn
        )
        self.experiment = dj.create_virtual_module(
            "experiment", "pipeline_experiment", connection=self.conn
        )
        self.mice = dj.create_virtual_module(
            "mice", "common_mice", connection=self.conn
        )
        self.lab = dj.create_virtual_module("lab", "common_lab", connection=self.conn)
        self.odor = dj.create_virtual_module(
            "odor", "pipeline_odor", connection=self.conn
        )

    @staticmethod
    def rgetattr(obj, attribute: str):
        """
        Recursively gets attributes of Python object. Used to get schema tables from a
        string describing it.

                Parameters:
                    obj (object): A schema or table object
                    attribute (str): The string name of a table or part table

                Returns:
                    obj (object): The specified table object

                Ex.
                    >>>my_server = DJServer('my_url.database.com', 'username', 'password123')
                    >>>my_server.rgetattr(my_server, 'meso.ScanInfo.Field')
                    return: datajoint.user_tables.Field
        """
        attributes = attribute.split(".")
        for attr in attributes:
            obj = getattr(obj, attr)
        return obj

    @staticmethod
    def copy_tables(source_server, target_server, key: dict, tables: tuple):
        """
        Copies all data which matches the given key from the tables in source_server
        to the target_server if it is not already present.

            WARNING:
                Target table keys data must be downloaded to estimate number of keys
                to be inserted and to avoid overwriting data. Running without narrow
                key restrictions will use a lot of memory and bandwidth.

            Parameters:
                source_server (DJServer): Instance which data is downloaded from
                target_server (DJServer): Instance which data is inserted into
                key (dict): Key which picks out data to download
                tables (tuple): Tuple of strings which defines tables to use

            Ex.
                >>>at_server = DJServer('at_url.database.com', 'username1', 'password123')
                >>>jr_server = DJServer('jr_url.database.com', 'username2', 'password456')
                >>>tables = ('meso.ScanInfo', 'meso.ScanInfo.Field')
                >>>DJServer.copy_tables(at_server, jr_server, {'animal_id': 17797}, tables)
        """
        print(f"Estimating data usage for transferring {key}...\n")
        num_new_keys = 0
        for table_str in tables:

            source_table = source_server.rgetattr(source_server, table_str)
            target_table = target_server.rgetattr(target_server, table_str)

            ## Check if the key has any attributes in common with the table
            key_and_table_matches = np.any(
                [k in source_table.primary_key for k in key.keys()]
            )

            ## Use key restriction if possible. Else, download/insert entire table.
            ## We need to fetch the target keys because cross-connection restrictions don't work.
            if key_and_table_matches:
                target_keys_already_present = (target_table & key).fetch("KEY")
                num_source_keys = len((source_table & key))
                num_uninserted_source_keys = len(
                    (source_table & key) - target_keys_already_present
                )
            else:
                target_keys_already_present = (target_table).fetch("KEY")
                num_source_keys = len((source_table))
                num_uninserted_source_keys = len(
                    source_table - target_keys_already_present
                )
            print(
                f"{num_uninserted_source_keys}/{num_source_keys} keys available for transfer into {table_str}"
            )
            num_new_keys += num_uninserted_source_keys

        print(f"\nAbout to insert {num_new_keys} keys. Proceed? (Y/N)")
        response = input()

        if response.lower() in ("y", "yes"):

            print("")
            for table_str in tables:

                print(f"Downloading data from {table_str}...")
                source_table = source_server.rgetattr(source_server, table_str)
                target_table = target_server.rgetattr(target_server, table_str)

                ## Check if the key has any attributes in common with the table
                key_and_table_matches = np.any(
                    [k in source_table.primary_key for k in key.keys()]
                )

                ## Use key restriction if possible. This reduces the amount of data downloaded.
                ## We need to fetch the target keys because cross-connection restrictions don't work.
                if key_and_table_matches:
                    target_keys_already_present = (target_table & key).fetch("KEY")
                    source_data = (
                        (source_table & key) - (target_keys_already_present)
                    ).fetch(as_dict=True)
                else:
                    target_keys_already_present = (target_table).fetch("KEY")
                    source_data = (source_table - target_keys_already_present).fetch(
                        as_dict=True
                    )

                ## Insert missing data and allow direct insert into dj.Computed tables
                target_table.insert(
                    source_data, allow_direct_insert=True, skip_duplicates=True
                )
                print(f"Inserted data for {table_str}!\n")

            print(f"Finished copying data for {key}!")

    ## These class variables will be used for helpful batch functions
    experiment_tables = (
        "mice.Mice",
        "experiment.Session",
        "experiment.Session.Fluorophore",
        "experiment.Session.PMTFilterSet",
        "experiment.Session.TargetStructure",
        "experiment.Scan",
        "experiment.Scan.BehaviorFile",
        "experiment.Scan.EyeVideo",
        "experiment.Scan.Laser",
        "experiment.ScanProtocol",
    )

    pupil_tables = (
        "pupil.Eye",
        "pupil.Tracking",
        "pupil.Tracking.Deeplabcut",
        "pupil.FittedPupil",
        "pupil.FittedPupil.Ellipse",
        "pupil.FittedPupil.EyePoints",
        "pupil.FittedPupil.Circle",
        "pupil.ProcessedPupil",
        "pupil.PupilUnitConversion",
        "pupil.PupilPeriods",
        "pupil.PupilPeriods.DilationConstriction",
    )

    treadmill_tables = (
        "treadmill.Treadmill",
        "treadmill.Running",
        "treadmill.Running.Period",
    )

    odor_tables = (
        "odor.Odorant",
        "odor.OdorSolution",
        "odor.OdorSession",
        "odor.OdorConfig",
        "odor.OdorRecording",
        "odor.OdorTrials",
        "odor.OdorSync",
        "odor.MesoMatch",
        "odor.Respiration",
    )

    scan_tables = (
        "meso.ScanInfo",
        "reso.ScanInfo",
        "meso.ScanInfo.Field",
        "reso.ScanInfo.Field",
        "meso.FieldAnnotation",
        "reso.FieldAnnotation",
        "meso.CorrectionChannel",
        "reso.CorrectionChannel",
        "meso.RasterCorrection",
        "reso.RasterCorrection",
        "meso.Quality",
        "reso.Quality",
        "meso.Quality.Contrast",
        "reso.Quality.Contrast",
        "meso.Quality.EpileptiformEvents",
        "reso.Quality.EpileptiformEvents",
        "meso.Quality.MeanIntensity",
        "reso.Quality.MeanIntensity",
        "meso.Quality.QuantalSize",
        "reso.Quality.QuantalSize",
        "meso.Quality.SummaryFrames",
        "meso.Quality.SummaryFrames",
    )


def __connect_at_jr_servers(
    at_url="at-database.ad.bcm.edu",
    at_user=None,
    at_pass=None,
    jr_url="jr-database.ad.bcm.edu",
    jr_user=None,
    jr_pass=None,
    **kwargs,
):
    """
    Returns Tolias and Reimer DJServer instances based on provided settings. This
    is a utility function not meant to be used directly by the user.

        Parameters:
            at_url (str) : String defining Tolias database URL
            at_user (str): String of Tolias database username. User will be prompted
                           with an input if no value is given.
            at_pass (str): String of Tolias database password. User will be prompted
                           with a password input if no value is given.
            jr_url (str) : String defining Reimer database URL
            jr_user (str): String of Reimer database username. User will be prompted
                           with an input if no value is given.
            jr_pass (str): String of Reimer database password. User will be prompted
                           with a password input if no value is given."""
    if at_user is None:
        at_user = input("Enter Tolias database username:\n")
    if at_pass is None:
        at_pass = getpass("Enter Tolias database password:\n")
    if jr_user is None:
        jr_user = input("Enter Reimer database username:\n")
    if jr_pass is None:
        jr_pass = getpass("Enter Reimer database password:\n")

    at_server = DJServer(at_url, at_user, at_pass)
    jr_server = DJServer(jr_url, jr_user, jr_pass)

    return at_server, jr_server


def copy_at_to_jr_tables(
    key: dict, tables: tuple, at_user: dict = None, jr_user: dict = None, **kwargs
):
    """
    Connects to Tolias and Reimer database, then transfers all data matching the
    provided key in the given tables from the Tolias to Reimer database if it is
    not already present.

        Parameters:
            key (dict)    : Key which defines relevant data to copy over
            tables (tuple): Tuple of strings, each string defining a specific table
                            to copy data from/into.
            at_user (str) : String of Tolias database username. User will be prompted
                            with an input if no value is given.
            jr_user (str) : String of Reimer database username. User will be prompted
                            with an input if no value is given.

        WARNING:
            The tables tuple transfers data in the order provided. Therefore, be sure any
            table dependencies are listed in the tuple before their downstream table.

            eg. tables=('meso.CorrectionChannel', 'meso.ScanInfo') will fail because keys
                in meso.CorrectionChannel must have a corresponding key in meso.ScanInfo

        Ex.
            >>>tables = ('meso.ScanInfo', 'meso.ScanInfo.Field')
            >>>key = {'animal_id': 17797, 'session': 5}
            >>>copy_at_to_jr_tables(key=key, tables=tables, at_user='username1', jr_user='username2')
    """

    at_server, jr_server = __connect_at_jr_servers(
        at_user=at_user, jr_user=jr_user, **kwargs
    )
    DJServer.copy_tables(
        source_server=at_server, target_server=jr_server, key=key, tables=tables
    )


def copy_experiment_data(
    key: dict, at_user: dict = None, jr_user: dict = None, **kwargs
):
    """
    Utility function to copy a subset of experiment tables from the Tolias
    to Reimer database matching the given key. See copy_at_to_jr_tables
    for full documentation.
    """
    tables_to_copy = DJServer.experiment_tables
    copy_at_to_jr_tables(
        key=key, tables=tables_to_copy, at_user=at_user, jr_user=jr_user, **kwargs
    )


def copy_pupil_data(key: dict, at_user: dict = None, jr_user: dict = None, **kwargs):
    """
    Utility function to copy a subset of pupil tables from the Tolias
    to Reimer database matching the given key. See copy_at_to_jr_tables
    for full documentation.
    """
    tables_to_copy = DJServer.pupil_tables
    copy_at_to_jr_tables(
        key=key, tables=tables_to_copy, at_user=at_user, jr_user=jr_user, **kwargs
    )


def copy_treadmill_data(
    key: dict, at_user: dict = None, jr_user: dict = None, **kwargs
):
    """
    Utility function to copy a subset of treadmill tables from the Tolias
    to Reimer database matching the given key. See copy_at_to_jr_tables
    for full documentation.
    """
    tables_to_copy = DJServer.treadmill_tables
    copy_at_to_jr_tables(
        key=key, tables=tables_to_copy, at_user=at_user, jr_user=jr_user, **kwargs
    )


def copy_odor_data(key: dict, at_user: dict = None, jr_user: dict = None, **kwargs):
    """
    Utility function to copy a subset of odor tables from the Tolias
    to Reimer database matching the given key. See copy_at_to_jr_tables
    for full documentation.
    """
    tables_to_copy = DJServer.odor_tables
    copy_at_to_jr_tables(
        key=key, tables=tables_to_copy, at_user=at_user, jr_user=jr_user, **kwargs
    )


def copy_scan_data(key: dict, at_user: dict = None, jr_user: dict = None, **kwargs):
    """
    Utility function to copy a subset of meso and reso tables from the Tolias
    to Reimer database matching the given key. See copy_at_to_jr_tables for
    full documentation.
    """
    tables_to_copy = DJServer.scan_tables
    copy_at_to_jr_tables(
        key=key, tables=tables_to_copy, at_user=at_user, jr_user=jr_user, **kwargs
    )