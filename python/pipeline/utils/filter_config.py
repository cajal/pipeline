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
    filter_hash                         :varchar(32)
    ---
    filter_name="online_median"         : varchar(16)           # filter_name
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
# class FilterConfig(dj.Lookup):
#     definition = """
#     # Filter configuration
#     filter_hash: varchar(32)
#     ---
#     """

#     @property
#     def contents(self):
#         num_layers = [1]
#         hidden_state = [8, 16]
#         input_kernel = [11]
#         hidden_kernel_size = [1]
#         regularizer_type = ['LaplaceL2']
#         regularizer_strength = [0.0, 2.0, 5.0, 10.0, 20.0, 50.0]

#         for p in product(hidden_state, input_kernel, hidden_kernel_size,
#                          num_layers, regularizer_type, regularizer_strength):

#             k = dict(zip(self.heading.dependent_attributes, p))
#             k['core_config_hash'] = key_hash(k)
#             yield k
