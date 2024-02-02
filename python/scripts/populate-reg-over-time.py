#!/usr/local/bin/python3

'''
For populating RegistrationOverTime
Prerequisite: 
    - insert scans and stacks to RegistrationTask
    - run populate-minion.py (this is usually run by the pipeline minion automatically)
    - insert scans and stacks of interest into stack.ZDriftQuery
'''

from pipeline import stack
import logging
import datajoint as dj

## database logging code 

logging.basicConfig(level=logging.ERROR)
logging.getLogger('datajoint.connection').setLevel(logging.DEBUG)
if hasattr(dj.connection, 'query_log_max_length'):
    dj.connection.query_log_max_length = 3000 

# delete errors
err_msg_timeout = 'error_message = "InternalError: (1205, \'Lock wait timeout exceeded; try restarting transaction\')"'
err_msg_sigterm = 'error_message = "SystemExit: SIGTERM received"'
timestamp = 'timestamp > "2021-12-10"'
(stack.schema.jobs & [err_msg_timeout, err_msg_sigterm] & timestamp).delete()

# populate RegistrationOverTime
stack.RegistrationOverTime.populate(reserve_jobs=True, suppress_errors=True)
