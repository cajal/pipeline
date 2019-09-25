import datajoint as dj
from pipeline import experiment

schema = dj.schema('pipeline_collection', create_tables=True)


@schema
class Study(dj.Manual):
    definition = """
    # class of experiments towards specific aim
    study_name                          :varchar(50)            # name of study 
    ---
    description                         :varchar(450)           # brief description of experimental details and study goals
    """


@schema
class CurationPurpose(dj.Lookup):
    definition = """
    # Targeted use for scans added to CuratedScans
    scan_purpose                        :varchar(100)           # scan use
    ---
    scan_criteria                       :varchar(450)           # brief description of properties scan must have
    purpose_ts = CURRENT_TIMESTAMP      :timestamp              # automatic at time of first entry
    """


@schema
class CuratedScan(dj.Manual):
    definition = """
    # Collection of scans approved for specified use
    ->Study
    ->experiment.Scan
    ->CurationPurpose
    score_ts = CURRENT_TIMESTAMP     :timestamp              # timestamp of score entry
    ---
    ->experiment.Person
    score				                :tinyint		        # subjective score of scan by experimenter
    notes=null                          :varchar(450)           # reviewer comments/concerns at time of entry
    """

@schema
class CuratedStack(dj.Manual):
    definition = """
    # Collection of stacks approved for specified use
    ->Study
    ->experiment.Stack
    ->CurationPurpose
    score_ts = CURRENT_TIMESTAMP     :timestamp              # timestamp of score entry
    ---
    ->experiment.Person
    score				                :tinyint		        # subjective score of stack by experimenter
    notes=null      	                :varchar(450)		    # reviewer comments/concerns at time of entry
    """