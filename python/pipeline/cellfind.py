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

schema = dj.schema('wenbo_celldetection', locals())
schema_cf = dj.schema('wenbo_cellfinder', locals())

@schema
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
    def populated_from(self):
        return pre.AverageFrame() & dict(channel = 1)

    # you should first click pts, the n enter yes to delete data
    def update_cell_locations(self, key):
        # Fetch frame
        assert pre.AverageFrame() & key, "key {0} is not in the table".format(str(key))
        ave_frame = (pre.AverageFrame() & key).fetch1['frame']
        mon_frame = (monet.Cos2Map() & key & 'ca_kinetics = 1').fetch1['cos2_amp']

        # Fetch existing points
        count = 1
        old_points = []
        fig, ax = None, None
        if self & key:
            x, y, count_array, fig, ax= self.mark_all(key)
            count = len(x) + 1
            old_points = np.c_[x,y, count_array].tolist()

        # Get new points
        new_point = image_preprocessing.mark.App(ave_frame, mon_frame, count, old_points, fig, ax)

        # Update cell locations information
        (self & key).delete()
        plt.show()
        plt.ioff()
        if not self & key:
            self.insert1(key)
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
        x_array, y_array, count_array =  (self.Location() & key).fetch['x','y', 'cell_id']

        # Plot
        plot_params = dict(cmap=plt.cm.get_cmap('bwr'))
        sns.set_style('white')
        fig, ax = plt.subplots()
        ax.imshow(frame, **plot_params)
        ax.scatter(x_array, y_array, c='red')
        ax.set_ylim((0, frame.shape[0]))
        ax.set_xlim((0, frame.shape[1]))
        plt.ion()
        plt.show()

        return x_array, y_array, count_array, fig, ax



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
        for i, (q, l, p) in enumerate(itertools.product([4, 8, 12], [4, 8, 12], ['LH', 'L'])):
        # for i, (q, l, p) in enumerate(itertools.product([12], [12], ['LH', 'L'])):

            yield i, q, l, p

@schema_cf
class TestSplit(dj.Computed):
    definition = """
    # holds different splits in to test and training sets
    ->rf.Session
    split_id        : tinyint   # id of the split
    ---
    """

    @property
    def populated_from(self):
        return (rf.Session() & CellLocations()).project()

    def _make_tuples(self, key):
        all_cell = np.array((CellLocations() & key).project().fetch.as_dict())
        chunksize = 3
        count = 1       # Accumulator for  split_id

        kf = KFold(len(all_cell), n_folds=len(all_cell) // chunksize)
        for train, test in kf:
            key['split_id'] = count
            self.insert1(key)
            for test_set in all_cell[test]:
                self.TestFrame().insert1(self.add_split_id(test_set, count))
            for train_set in all_cell[train]:
                self.TrainingFrame().insert1(self.add_split_id(train_set, count))
            count += 1

    def add_split_id(self, set, id):
        set['split_id'] = id
        return set

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
class ModelSelection(dj.Computed):
    definition = """
    -> TestSplit
    ---
    -> Parameters
    test_score  : float # average test score over test splits
    """

    class ModelParameters(dj.Part):
        definition = """
        # parameters of best model

        ->ModelSelection
        ---
        u_xy        : longblob  # quadratic filters
        w_xy        : longblob  # linear filters
        beta        : longblob  # beta for quadratic
        gamma       : longblob  # gamma for linear
        b           : longblob  # the constant
        """

    class ParametersScore(dj.Part):
        definition = """
        #score for each parameter for every split

        -> ModelSelection
        -> Parameters
        ---
        score       : float # score for each parameter
        """

    def _make_tuples(self, key):
        print("up to date now new new before leave")
        best_score_dict = dict()
        chunksize, filter_size = 3, 9
        score_key = copy.deepcopy(key)
        self.fetch_frame(key)
        conn = dj.conn()

        # For a specific split combination, we iterate through every parameter combination on it.
        for param in Parameters().fetch.as_dict:
            param_score = []
            kf = KFold(len(self.train_frame), n_folds=len(self.train_frame) // chunksize)

            # Get the score for each parameter.
            for train, val in kf:
                validation_stack, validation_loc = self.train_frame[val], self.train_loc[val]
                train_stack, train_loc = self.train_frame[train], self.train_loc[train]
                val_result, _ = self.train_fuc(validation_stack, train_stack, validation_loc, train_loc, param, filter_size)
                param_score.append(val_result)
                print('connected : {}'.format(conn.is_connected))
            param_result = reduce(lambda x, y: x + y, param_score) / float(len(param_score))
            best_score_dict[param['param_id']] = param_result
            print("key : {}, param_id : {}, score : {}".format(key, param['param_id'], param_result))

        # Find out the parameter that gives best score.
        best_param_id = max(best_score_dict, key = (lambda key : best_score_dict[key]))
        print("best param id : {}".format(best_param_id))
        best_comb = (Parameters() & dict(param_id = best_param_id)).fetch.as_dict()[0]
        # Use the best parameter to test the test_set
        f_r, tdbinstance = self.train_fuc(self.test_frame, self.train_frame, self.test_loc, self.train_loc, best_comb, filter_size)
        key['param_id'], key['test_score']= best_param_id, f_r
        self.insert1(key)

        # Record the score of each parameter for a specific split id.
        for i_key in best_score_dict:
            score_key['param_id'], score_key['score'] = i_key, best_score_dict[i_key]
            self.ParametersScore().insert1(score_key)

        # Record the value of the best parameter.
        subkey = dict()
        subkey['animal_id'], subkey['session'], subkey['split_id'], subkey['u_xy'], subkey['w_xy'], subkey['beta'], \
        subkey['gamma'], subkey['b'] = key['animal_id'], key['session'], key['split_id'], tdbinstance.parameters[
            'u_xy'], tdbinstance.parameters['w_xy'], tdbinstance.parameters['beta'], tdbinstance.parameters['gamma'], \
                                       tdbinstance.parameters['b']
        self.ModelParameters().insert1(subkey)

    def fetch_frame(self, key):
        '''
        Build test set's frame and cell locations for a given split combination;
        Build training set's frame and cell locations for a given split combination.
        :param key: a valid key for TestSplit() Table
        '''

        # Build frame lists
        self.test_frame = ((TestSplit().TestFrame() & key) * pre.AverageFrame()).fetch['frame']
        self.train_frame = ((TestSplit().TrainingFrame() & key) * pre.AverageFrame()).fetch['frame']

        # Build cell_locations list.
        self.test_loc, self.train_loc = [], []
        for testkey in (TestSplit().TestFrame() & key).fetch.as_dict():
            gene_cell_loc(testkey, self.test_loc)
        self.test_loc = np.array(self.test_loc)
        for trainkey in (TestSplit().TrainingFrame() & key).fetch.as_dict():
            gene_cell_loc(trainkey, self.train_loc)
        self.train_loc = np.array(self.train_loc)

    def train_fuc(self, v_stack, t_stack, v_loc, t_loc, param, filter_size):
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
            v_stack[validation_idx] = gene_stack((filter_size, filter_size), v_stack[validation_idx], param['preprocessing'])
        for train_idx in range(len(t_stack)):
            t_stack[train_idx] = gene_stack((filter_size, filter_size), t_stack[train_idx], param['preprocessing'])

        # Initialize the tdb instance and train on the training set.
        c = bernoulli_process.RDBP((filter_size, filter_size), linear_channels=param['linear'], exponentials=min(param['linear'], param['quadratic']), quadratic_channels=param['quadratic'])
        c.fit(list(t_stack), list(t_loc))

        # Test on the testing sets and return the average score
        result_list = []
        for entry in zip(v_stack, v_loc):
            result_list.append(c.auc(entry[0], entry[1]))
        mean = reduce(lambda x, y: x + y, result_list) / float(len(result_list))
        return mean, c


    def to_num(self, key):
        return tuple(key.values())


def gene_cell_loc(key, cell_list):
    '''
    :param key: a valid key for CellLocations().Location()
    :param cell_list: a list holding all the cell locations for a stack
    '''

    locs = (CellLocations().Location() & key).fetch['x', 'y']
    cell_location = np.array(list(zip(locs[0], locs[1]))).astype(int)
    cell_list.append(cell_location)


def gene_stack(filtersize, frame, preoption):
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

    newframe_shape = tuple(i + j -1 for i,j in zip(frame.shape, filtersize))
    newframe = np.ones(newframe_shape)

    i, j = [(i - j + 1) // 2 for i, j in zip(newframe.shape, frame.shape)]
    newframe[i:-i, j:-j] = frame
    return newframe
