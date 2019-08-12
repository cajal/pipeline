import datajoint as dj
import numpy as np
import hashlib
from itertools import product

schema = dj.schema('pipeline_eye', locals())


def key_hash(key):
    """
    32-byte hash used for lookup of primary keys of jobs
    """
    hashed = hashlib.md5()
    for k, v in sorted(key.items()):
        hashed.update(str(v).encode())
    return hashed.hexdigest()


@schema
class OnlineMedianFilter(dj.Lookup):
    definition = """
    # online median filter parameters
    filter_hash                         : varchar(32)
    ---
    filter_name="online_median"         : varchar(16)           # filter name
    kernel_size                         : tinyint unsigned      # kernel size
    """

    @property
    def contents(self):
        filter_name = ["online_median"]
        kernel_size = [3, 5, 7, 9]

        for p in product(filter_name, kernel_size):
            k = dict(zip(self.heading.dependent_attributes,p))
            k['filter_hash'] = key_hash(k)
            yield k

# @schema
# class LowspassFilter(dj.Lookup):
#     definition = """
#     # lowpass filter parameters
#     filter_hash                         : varchar(32)
#     ---
#     filter_name="lowpass"               : varchar(16)           # filter name
#     sampling_freq                       : float                 # sampling freq
#     cutoff_freq                         : float                 # cutoff freq
#     """

#     @property
#     def contents(self):
#         filter_name = ["lowpass"]
#         sampling_freq = [20.0]
#         cutoff_freq = [5.0, 7.0, 9.0]

#         for p in product(filter_name, sampling_freq, cutoff_freq):
#             k = dict(zip(self.heading.dependent_attributes,p))
#             k['filter_hash'] = key_hash(k)
#             yield k

# @schema
# class FilterTable(dj.)