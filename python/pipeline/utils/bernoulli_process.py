import numpy as np
import theano as th
from collections import OrderedDict
from scipy.optimize import minimize
T = th.tensor
import theano.tensor.nnet.conv3d2d
from sklearn.metrics import roc_auc_score
from theano import shared
th.config.floatX = 'float32'
floatX = th.config.floatX


class BernoulliProcess:
    def __init__(self, pixel, exponentials=4, linear_channels=4, quadratic_channels=4):
        if np.any(np.array(pixel) % 2 == 0):
            raise ValueError("Pixel size should be odd.")
        self.pixel = pixel

        # self.parameters = OrderedDict()
        assert linear_channels > 0, "should use at least one linear channel."
        assert quadratic_channels > 0, "should use at least one quadratic channel."


        self.linear_channels = linear_channels
        self.quadratic_channels = quadratic_channels
        self.common_channels = exponentials

        flt_width, flt_height = self.pixel

        # horizontal components of the filters
        self.u = shared(value=np.random.rand(quadratic_channels, flt_width, flt_height).astype(floatX) / (quadratic_channels*flt_width*flt_height), name='u', borrow=True)
        # horizontal components of the filters
        self.w = shared(value=np.random.rand(linear_channels, flt_width, flt_height).astype(floatX) / (linear_channels*flt_width*flt_height), name='w', borrow=True)

        self.beta = shared(value=np.random.randn(exponentials, quadratic_channels).astype(floatX), name='beta', borrow=True)
        self.gamma = shared(value=np.random.randn(exponentials, linear_channels).astype(floatX), name='gamma', borrow=True)
        self.b = shared(value=np.random.randn(exponentials).astype(floatX), name='b', borrow=True)


    def _build_label_stack(self, data_shape, cell_locations):
        """
        Builds that stack in which the locations indicated by cell_locations are set to one.
        Otherwise the values of the stack are zero.
        :param data_shape: shape of original stack
        :param cell_locations: Nx2 integer array will cell locations (0 based indices)
        :param full: indicates whether the result should have full size of the size after valid convolution
        :return: numpy array with stack indicating the cell locations.
        """
        y_shape = tuple(i - j + 1 for i, j in zip(data_shape, self.pixel))
        Y = np.zeros(y_shape, dtype=floatX)
        j, i = cell_locations.T
        Y[i, j] = 1

        return Y

    def _single_cross_entropy(self, data_shape, learning_rate):
        X_ = T.tensor3('stack', dtype=floatX) # no images x 1 x row x col
        Y_ = T.tensor3('cells', dtype=floatX) # no images x row x col

        in_channels = 1
        batchsize, in_row, in_col = data_shape
        flt_row, flt_col = self.pixel

        quadratic_features_ = T.nnet.conv2d(
            # expects (batch size, channels, row, col)
            input=X_.dimshuffle(0,'x',1,2),
            # expects nb filters, channels, nb row, nb col
            filters=self.u.dimshuffle(0,'x',1,2),
            filter_shape=(self.u.get_value(borrow=True).shape[0], in_channels, flt_row, flt_col),
            input_shape=(batchsize, in_channels, in_row, in_col),
            border_mode='valid',
        )   #(15,3,256,256)

        linear_features_ = T.nnet.conv2d(
            # expects (batch size,  channels, row, col)
            input=X_.dimshuffle(0,'x',1,2),
            # expects nb filters, channels, nb row, nb col
            filters=self.w.dimshuffle(0,'x',1,2),
            filter_shape=(self.w.get_value(borrow=True).shape[0], in_channels, flt_row, flt_col),
            input_shape=(batchsize, in_channels, in_row, in_col),
            border_mode='valid',
        )   #(15,3,256,256)

        quadr_filter_ = T.tensordot(self.beta, quadratic_features_ ** 2, (1,1)) #4d (2,15,256,256)
        lin_filter_ = T.tensordot(self.gamma, linear_features_, (1, 1)) #4d (2,15,256,256)

        exponent_ = quadr_filter_ + lin_filter_ + self.b.dimshuffle(0, 'x', 'x','x')

        p_ = T.exp(exponent_).sum(axis=0)
        p2_ = p_ / (1 + p_) * (1 - 2 * 1e-8) + 1e-8  #3d (15,254,254)
        loglik_ = Y_ * T.log(p2_) + (1 - Y_) * T.log(1 - p2_)
        cross_entropy_ = -T.mean(loglik_)
        g_u_ = T.grad(cross_entropy_, self.u)
        g_w_ = T.grad(cross_entropy_, self.w)
        g_beta_ = T.grad(cross_entropy_, self.beta)
        g_gamma_ = T.grad(cross_entropy_, self.gamma)
        g_b_ = T.grad(cross_entropy_, self.b)

        updates = [(self.u, self.u - learning_rate * g_u_),
                   (self.w, self.w - learning_rate * g_w_),
                   (self.beta, self.beta - learning_rate * g_beta_),
                   (self.gamma, self.gamma - learning_rate * g_gamma_),
                   (self.b, self.b - learning_rate * g_b_)]

        train_model = th.function(inputs=[X_, Y_], outputs=cross_entropy_, updates=updates)

        # f = th.function([X_], linear_features_, allow_input_downcast=True)
        #
        # import numpy as np
        # X = np.random.randn(15,256,256).astype('float32')
        # Y = np.random.randint(2, size=(15,254,254))
        # fx = f(X)
        # #----------------------
        # # TODO: remove later
        # from IPython import embed
        # embed()
        # #exit()
        # #----------------------

        return train_model

    def _build_p_map(self, data_shape, X_):

        in_channels = 1
        batchsize, in_row, in_col = data_shape
        flt_row, flt_col = self.pixel

        quadratic_features_ = T.nnet.conv2d(
            # expects (batch size, channels, row, col)
            input=X_.dimshuffle(0, 'x', 1, 2),
            # expects nb filters, channels, nb row, nb col
            filters=self.u.dimshuffle(0, 'x', 1, 2),
            filter_shape=(self.u.get_value(borrow=True).shape[0], in_channels, flt_row, flt_col),
            input_shape=(batchsize, in_channels, in_row, in_col),
            border_mode='valid',
        )  # (15,3,256,256)

        linear_features_ = T.nnet.conv2d(
            # expects (batch size,  channels, row, col)
            input=X_.dimshuffle(0, 'x', 1, 2),
            # expects nb filters, channels, nb row, nb col
            filters=self.w.dimshuffle(0, 'x', 1, 2),
            filter_shape=(self.w.get_value(borrow=True).shape[0], in_channels, flt_row, flt_col),
            input_shape=(batchsize, in_channels, in_row, in_col),
            border_mode='valid',
        )  # (15,3,256,256)

        quadr_filter_ = T.tensordot(self.beta, quadratic_features_ ** 2, (1, 1))  # 4d (2,15,256,256)
        lin_filter_ = T.tensordot(self.gamma, linear_features_, (1, 1))  # 4d (2,15,256,256)

        exponent_ = quadr_filter_ + lin_filter_ + self.b.dimshuffle(0, 'x', 'x', 'x')

        p_ = T.exp(exponent_).sum(axis=0)
        p2_ = p_ / (1 + p_) * (1 - 2 * 1e-8) + 1e-8  # 3d (15,254,254)

        return p2_

    def P(self, X):
        X_ = T.tensor3('stack_P', dtype=floatX)  # th.shared(np.require(Y, dtype=floatX), borrow=True, name='cells')
        p_ = self._build_p_map(X.shape, X_)
        p = th.function([X_], p_, allow_input_downcast=True)
        P = p(X)

        return P

    def auc(self, X, cell_locations, **kwargs):
        """
        Computes the area under the curve for the current parameter setting.
        :param X: stack
        :param cell_locations: N X 3 array of cell locations (dtype=int)
        :param kwargs: additionaly keyword arguments passed to sklearn.metrics.roc_auc_score
        :return: area under the curve
        """
        result_list = []
        i = 0
        p_array = self.P(X)
        for p_map in p_array:
            result_list.append(roc_auc_score(self._build_label_stack(X[i].shape, cell_locations[i]).ravel(), p_map.ravel(), **kwargs))
            i += 1
        return result_list

    def fit(self, stacks, cell_locations, learning_rate, grad_stacks, grad_locations, **options):
        """
        Fits the model.
        :param stacks: Iterable of stacks. All of them need to have the same size.
        :param cell_locations: Iterable of cell locations (N X 3 array of dtype int)
        :param options: options for scipy.minimize
        """
        print("stack shape : {}".format(stacks.shape))
        ll = self._single_cross_entropy(stacks.shape, learning_rate)
        Ys = [self._build_label_stack(x.shape, c) for x, c in zip(stacks, cell_locations)]
        Ys = np.asarray(Ys)

        pre_val_auc = 0
        iteration = 1
        # while True:
        for i in range(10001):            # print("parameter beta : {}".format(self.parameters['beta'].get_value()))
            # if iteration // 100 == 0:
            #     val_auc = self.auc(grad_stacks, grad_locations)
            #     print("validation auc result : {}".format(val_auc))
            #     if pre_val_auc > val_auc:
            #         break
            #     else:
            #         pre_val_auc = val_auc

            cross = ll(stacks, Ys)
            print("parameter beta : {}".format(self.beta.get_value(borrow=True)))
            print("{}: cross entropy {}".format(iteration, cross))
            iteration += 1



    def __str__(self):
        return """
        Range degenerate Bernoulli process
        quadratic components: %i
        linear components: %i
        common components: %i
        """ % (self.quadratic_channels, self.linear_channels, self.common_channels)


    def __repr__(self):
        return self.__str__()