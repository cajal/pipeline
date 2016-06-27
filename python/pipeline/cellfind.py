import copy
import datajoint as dj
import itertools
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from functools import reduce
from pipeline import rf, pre, monet
from pipeline.utils.image_preprocessing import local_standardize, histeq
from sklearn.cross_validation import KFold
from .utils import image_preprocessing, bernoulli_process

schema_cf = dj.schema('wenbo_cellfinder_test', locals())


@schema_cf
class CellLocations(dj.Imported):
    definition = """
    -> pre.AverageFrame
    ---
    """

    class Location(dj.Part):
        definition = """
        ->CellLocations
        cell_id     : smallint # cell identifier
        ---
        x           : float    # x coordinate of the cell
        y           : float    # y coordinate of the cell
        """

    def _make_tuples(self, key):
        self.update_cell_locations(key)

    @property
    def key_source(self):
        return pre.AverageFrame() & dict(channel=1)

    # you should first click pts, the n enter yes to delete data
    def update_cell_locations(self, key):
        # Fetch frame
        assert pre.AverageFrame() & key, "key {0} is not in the AverageFrame table".format(str(key))
        assert monet.Cos2Map() & key & 'ca_kinetics = 1', "key {0} is not in the Cos2Map table".format(str(key))
        ave_frame = (pre.AverageFrame() & key).fetch1['frame']
        mon_frame = (monet.Cos2Map() & key & 'ca_kinetics = 1').fetch1['cos2_amp']

        # Fetch existing points
        old_points = []
        fig, ax = None, None
        if self & key:
            x, y, fig, ax = self.mark_all(key)
            old_points = list(zip(x, y))

        # Get new points
        new_point = image_preprocessing.App(ave_frame, mon_frame, old_points, fig, ax)

        # Update cell locations information
        (self & key).delete()
        if not self & key:
            self.insert1(key)
        new_point.data_list = [n + (i,) for i, n in enumerate(new_point.data_list, start=1)]
        for entry in new_point.data_list:
            key['cell_id'], key['x'], key['y'] = entry[2], entry[0], entry[1]
            CellLocations.Location().insert1(key)

    def mark_all(self, key):
        '''
        Mark all the existing points
        :param key: a valid query key for AverageFrame()
        :return : all the x and y positions of cells; cell id; the plot itself
        '''
        assert pre.AverageFrame() & key, "key {0} is not in the table".format(str(key))

        # Fetch the unmarked frame and cell locations in that frame
        frame = (pre.AverageFrame() & key).fetch1['frame']
        x_array, y_array = (self.Location() & key).fetch['x', 'y']

        # Plot
        plot_params = dict(cmap=plt.cm.get_cmap('bwr'))
        sns.set_style('white')
        fig, ax = plt.subplots()
        ax.imshow(frame, **plot_params)
        ax.scatter(x_array, y_array, c='red')
        ax.set_ylim((0, frame.shape[0]))
        ax.set_xlim((0, frame.shape[1]))

        return x_array, y_array, fig, ax


@schema_cf
class Parameters(dj.Lookup):
    definition = """
    # parameter settings for model selections

    param_id        : tinyint # parameter set id
    ---
    quadratic       : tinyint # number of quadratic channels
    linear          : tinyint # number of linear channels
    preprocessing   : char(5) # preprocelling
    """

    @property
    def contents(self):
        # for i, (q, l, p) in enumerate(itertools.product([4, 8, 12], [4, 8, 12], ['LH', 'L'])):
        for i, (q, l, p) in enumerate(itertools.product([4], [4, 8], ['LH'])):
            yield i, q, l, p


@schema_cf
class Chunksize(dj.Lookup):
    definition = """
    # chunksize for model selections

    chunk_id       : tinyint # chunksize id
    ---
    chunk_size     : tinyint # chunksize
    """

    @property
    def contents(self):
        yield [1, 2]


@schema_cf
class Filtersize(dj.Lookup):
    definition = """
    # filtersize for model selections

    filtersize_id       : tinyint # filtersize id
    ---
    filter_size     : tinyint # filtersize
    """

    @property
    def contents(self):
        yield [1, 9]


@schema_cf
class TestSplit(dj.Computed):
    definition = """
    # holds different splits in to test and training sets
    ->rf.Session
    split_id        : tinyint   # id of the split
    ---
    """

    @property
    def key_source(self):
        return (rf.Session() & CellLocations()).proj()

    def _make_tuples(self, key):
        all_cell = np.array((CellLocations() & key).proj().fetch.as_dict()[:6])
        chunksize = Chunksize().fetch1()['chunk_size']

        numfold = len(all_cell) // chunksize
        assert numfold != 1, "number of folds can not equal to 1"

        for count, (train, test) in enumerate(KFold(len(all_cell), n_folds=numfold), start=1):
            key['split_id'] = count
            self.insert1(key)
            for test_set in all_cell[test]:
                self.TestFrame().insert1(dict(test_set, split_id=count))
            for train_set in all_cell[train]:
                self.TrainingFrame().insert1(dict(train_set, split_id=count))

    class TestFrame(dj.Part):
        definition = """
        -> TestSplit
        -> CellLocations
        ---
        """

    class TrainingFrame(dj.Part):
        definition = """
        -> TestSplit
        -> CellLocations
        ---
        """


@schema_cf
class ModelSelection2(dj.Computed):
    definition = """
    -> TestSplit
    ---
    -> Parameters
    test_score  : float # average test score over test splits
    """

    class ModelParameters(dj.Part):
        definition = """
        # parameters of best model

        ->ModelSelection2
        ---
        u_xy        : longblob  # quadratic filters
        w_xy        : longblob  # linear filters
        beta        : longblob  # beta for quadratic
        gamma       : longblob  # gamma for linear
        b           : longblob  # the constant
        """

    class ParametersScore(dj.Part):
        definition = """
        # Keep record of all scores for each parameter combination for every split

        -> ModelSelection2
        -> Parameters
        ---
        score       : float # score for each parameter
        """

    def _make_tuples(self, key):
        best_score_dict = dict()
        chunksize = Chunksize().fetch1()['chunk_size']
        filter_size = Filtersize().fetch1()['filter_size']
        score_key = dict(key)
        f_test_frame, f_train_frame, f_test_loc, f_train_loc = self.fetch_frame(key)
        conn = dj.conn()

        # For a specific split combination, we iterate through every parameter combination on it.
        for param in Parameters().fetch.as_dict:
            param_score = []

            numfold = len(f_train_frame) // chunksize
            assert numfold != 1, "number of folds can not equal to 1"

            # Get the score for each parameter.
            for train, val in KFold(len(f_train_frame), n_folds=numfold):
                validation_stack, validation_loc = f_train_frame[val], f_train_loc[val]
                train_stack, train_loc = f_train_frame[train], f_train_loc[train]
                val_result, _ = self.train_function(validation_stack, train_stack, validation_loc, train_loc, param,
                                                    filter_size)
                param_score.append(val_result)
                conn.is_connected
            param_result = np.mean(param_score)
            best_score_dict[param['param_id']] = param_result
            print("key : {}, param_id : {}, score : {}".format(key, param['param_id'], param_result))

        # Find out the parameter that gives best score.
        best_param_id = max(best_score_dict, key=lambda key: best_score_dict[key])
        print("best param id : {}".format(best_param_id))
        best_comb = (Parameters() & dict(param_id=best_param_id)).fetch1()
        # Use the best parameter to test the test_set
        f_r, tdbinstance = self.train_function(f_test_frame, f_train_frame, f_test_loc, f_train_loc, best_comb,
                                               filter_size)
        key['param_id'], key['test_score'] = best_param_id, f_r
        self.insert1(key)

        # Record the score of each parameter for a specific split id.
        for i_key in best_score_dict:
            score_key['param_id'], score_key['score'] = i_key, best_score_dict[i_key]
            self.ParametersScore().insert1(score_key)

        # Record the value of the best parameter.
        del key['param_id'], key['test_score']
        key.update(tdbinstance.parameters)
        self.ModelParameters().insert1(key)

    def fetch_frame(self, key):
        '''
        Build test set's frame and cell locations for a given split combination;
        Build training set's frame and cell locations for a given split combination.
        :param key: a valid key for TestSplit() Table
        '''

        # Build frame lists
        test_frame = ((TestSplit().TestFrame() * pre.AverageFrame()) & key).fetch['frame']
        train_frame = ((TestSplit().TrainingFrame() * pre.AverageFrame()) & key).fetch['frame']

        # Build cell_locations list.
        test_loc = generate_cell_loc((TestSplit().TestFrame() & key).fetch.as_dict())
        train_loc = generate_cell_loc((TestSplit().TrainingFrame() & key).fetch.as_dict())
        return test_frame, train_frame, test_loc, train_loc

    def train_function(self, v_stack, t_stack, v_loc, t_loc, param, filter_size):
        '''
         Given training and testing sets, train the data and test on the test set.
         Return the average score of the test set and the tdb instance.
        :param v_stack: frames of training set
        :param t_stack: frames of testing set
        :param v_loc: cell locations for training set
        :param t_loc: cell locations for testing set
        :param param: parameters information
        :param filter_size: filter size
        '''

        # Build the input train and test stacks from train and test sets.
        for validation_idx in range(len(v_stack)):
            v_stack[validation_idx] = generate_stack((filter_size, filter_size), v_stack[validation_idx],
                                                     param['preprocessing'])
        for train_idx in range(len(t_stack)):
            t_stack[train_idx] = generate_stack((filter_size, filter_size), t_stack[train_idx], param['preprocessing'])

        # Initialize the tdb instance and train on the training set.
        c = bernoulli_process.RDBP((filter_size, filter_size), linear_channels=param['linear'],
                                   exponentials=min(param['linear'], param['quadratic']),
                                   quadratic_channels=param['quadratic'])
        c.fit(list(t_stack), list(t_loc))

        # Test on the testing sets and return the average score
        result_list = []
        for entry in zip(v_stack, v_loc):
            result_list.append(c.auc(entry[0], entry[1]))
        mean = np.mean(result_list)
        return mean, c

    def to_num(self, key):
        return tuple(key.values())


def generate_cell_loc(iterdict):
    loc_list = []
    for key in iterdict:
        locs = (CellLocations().Location() & key).fetch['x', 'y']
        cell_location = np.array(list(zip(locs[0], locs[1]))).astype(int)
        loc_list.append(cell_location)
    return np.array(loc_list)


def generate_stack(filtersize, frame, preoption):
    '''

    :param filtersize:
    :param frame:
    :param preoption:
    :return:
    '''
    if preoption == 'LH':
        frame = local_standardize(histeq(frame))
    if preoption == 'L':
        frame = local_standardize(frame)

    newframe_shape = tuple(i + j - 1 for i, j in zip(frame.shape, filtersize))
    newframe = np.ones(newframe_shape)

    i, j = [(i - j + 1) // 2 for i, j in zip(newframe.shape, frame.shape)]
    newframe[i:-i, j:-j] = frame
    return newframe
