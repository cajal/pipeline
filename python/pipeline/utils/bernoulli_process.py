import numpy as np
from scipy.optimize import minimize
import theano as th
import datajoint as dj
from collections import OrderedDict

floatX = th.config.floatX
T = th.tensor
import theano.tensor.nnet.conv3d2d
from sklearn.metrics import roc_auc_score

tensor5 = theano.tensor.TensorType('float64', 5 * [False])  # 5-dimension tensor?


class BP:
    def __init__(self, pixel):
        if np.any(np.array(pixel) % 2 == 0):
            raise ValueError("Pixel size should be odd.")
        self.pixel = pixel
        self.parameters = OrderedDict()

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
        Y = np.zeros(y_shape)
        j, i = cell_locations.T
        Y[i, j] = 1

        return Y

    def _single_cross_entropy(self, data_shape):
        X_ = T.dmatrix('stack')  # th.shared(np.require(Y, dtype=floatX), borrow=True, name='cells')
        Y_ = T.dmatrix('cells')  # th.shared(np.require(Y, dtype=floatX), borrow=True, name='cells')
        p_, parameters_ = self._build_probability_map(X_, data_shape)

        loglik_ = Y_ * T.log(p_) + (1 - Y_) * T.log(1 - p_)
        cross_entropy_ = -T.mean(loglik_)

        dcross_entropy_ = T.grad(cross_entropy_, parameters_)

        return th.function((X_, Y_) + parameters_, cross_entropy_), \
               th.function((X_, Y_) + parameters_, dcross_entropy_)

    def P(self, X):
        X_ = T.dmatrix('stack')  # th.shared(np.require(Y, dtype=floatX), borrow=True, name='cells')
        p_, params_ = self._build_probability_map(X_, X.shape)
        p = th.function((X_,) + params_, p_)
        P = p(*((X,) + tuple(self.parameters.values())))

        return P

    def auc(self, X, cell_locations, **kwargs):
        """
        Computes the area under the curve for the current parameter setting.
        :param X: stack
        :param cell_locations: N X 3 array of cell locations (dtype=int)
        :param kwargs: additionaly keyword arguments passed to sklearn.metrics.roc_auc_score
        :return: area under the curve
        """
        return roc_auc_score(self._build_label_stack(X.shape, cell_locations).ravel(), self.P(X).ravel(), **kwargs)

    def fit(self, stacks, cell_locations, **options):
        """
        Fits the model.
        :param stacks: Iterable of stacks. All of them need to have the same size.
        :param cell_locations: Iterable of cell locations (N X 3 array of dtype int)
        :param options: options for scipy.minimize
        """
        ll, dll = self._single_cross_entropy(stacks[0].shape)
        Ys = [self._build_label_stack(x.shape, c) for x, c in zip(stacks, cell_locations)]
        slices, shapes = [], []
        i = 0
        for elem in self.parameters.values():
            slices.append(slice(i, i + elem.size))
            shapes.append(elem.shape)
            i += elem.size
        iteration = 0
        def ravel(params):
            return np.hstack([e.ravel() for e in params])

        def unravel(x):
            return tuple(x[sl].reshape(sh) for sl, sh in zip(slices, shapes))

        def obj(par):
            return np.mean([ll(*((x, y) + unravel(par))) for x, y in zip(stacks, Ys)])

        def dobj(par):
            return np.mean([ravel(dll(*((x, y) + unravel(par)))) for x, y in zip(stacks, Ys)], axis=0)

        def callback(x):
            nonlocal iteration
            print('%03i Cross entropy:' % iteration, obj(x))
            iteration += 1

        x0 = ravel(self.parameters.values())

        # todo find a better way than to box constrain the parameters
        opt_results = minimize(obj, x0, jac=dobj, method='L-BFGS-B', callback=callback,
                               bounds=list(zip(-100 * np.ones(len(x0)), 100 * np.ones(len(x0)))),
                               options={"maxiter": 200})
        for k, param in zip(self.parameters, unravel(opt_results.x)):
            self.parameters[k] = param


class RDBP(BP):
    def __init__(self, pixel, exponentials=2, linear_channels=2, quadratic_channels=2):
        super(RDBP, self).__init__(pixel)
        assert linear_channels > 0, "should use at least one linear channel."
        assert quadratic_channels > 0, "should use at least one quadratic channel."
        self.linear_channels = linear_channels
        self.quadratic_channels = quadratic_channels
        self.common_channels = exponentials
        flt_width, flt_height = self.pixel

        # horizontal components of the filters
        self.parameters['u_xy'] = np.random.rand(quadratic_channels, flt_width, flt_height)
        self.parameters['u_xy'] /= self.parameters['u_xy'].size

        # horizontal components of the filters
        self.parameters['w_xy'] = np.random.rand(linear_channels, flt_width, flt_height)
        self.parameters['w_xy'] /= self.parameters['w_xy'].size

        self.parameters['beta'] = np.random.randn(exponentials, quadratic_channels)  # Beta for quadratic
        self.parameters['gamma'] = np.random.randn(exponentials, linear_channels)  # Gamma for linear
        self.parameters['b'] = np.random.randn(exponentials)  # The constant

    def _build_separable_convolution(self, no_of_filters, X_, data_shape):
        """
        Builds a theano function that performas a 2d convolution which is separable in
        xy vs. z on the stack.
        :param no_of_filters: number of convolution filters
        :param X_: 3d tensor representing the stack (row, col, depth (# of 2D images we want to process at once. ))
        :param data_shape: shape of the real data (the tensor has no shape yet)
        :return: theano symbolic expression, (Uxy tensor, Uz tensor)
        """
        Vxy_ = T.tensor3(dtype=floatX)
        batchsize, in_channels = 1, 1
        in_width, in_height = data_shape
        flt_row, flt_col = self.pixel

        xy_ = T.nnet.conv2d(
            # expects (batch size, channels, row, col), transform in to (depth(# of 2D images), 1, row, col)
            input=X_.dimshuffle('x', 'x', 0, 1),
            # expects nb filters, channels, nb row, nb col
            filters=Vxy_.dimshuffle(0, 'x', 1, 2),
            filter_shape=(no_of_filters, in_channels, flt_row, flt_col),
            image_shape=(batchsize, in_channels, in_width, in_height),
            border_mode='valid'
        ).dimshuffle(1, 2, 3)
        return xy_, Vxy_

    def _build_exponent(self, X_, data_shape):
        """
        Builds the exponent of the nonlinearty (see README or Theis et al. 2013)
        :param X_: 3d tensor representing the stack (row, col, depth (# of 2D images we want to include in a stack. ))
        :param data_shape: shape of the real data (the tensor has no shape yet)
        :return: symbolic tensor for the exponent, (Uxy, Wxy, beta, gamma, b)
        """

        linear_channels, quadratic_channels, common_channels = \
            self.linear_channels, self.quadratic_channels, self.common_channels

        quadratic_filter_, Uxy_ = self._build_separable_convolution(quadratic_channels, X_, data_shape)
        linear_filter_, Wxy_ = self._build_separable_convolution(linear_channels, X_, data_shape)
        b_ = T.dvector()
        beta_ = T.dmatrix()
        gamma_ = T.dmatrix()

        quadr_filter_ = T.tensordot(beta_, quadratic_filter_ ** 2, (1, 0))
        lin_filter_ = T.tensordot(gamma_, linear_filter_, (1, 0))

        exponent_ = quadr_filter_ + lin_filter_ + b_.dimshuffle(0, 'x','x')
        return exponent_, (Uxy_, Wxy_, beta_, gamma_, b_)

    def _build_probability_map(self, X_, data_shape):
        """
        Builds a theano symbolic expression that yields the estimated probability of a cell per voxel.
        :param X_: 3d tensor representing the stack (row, col, depth(possible: # of 2D images in a stack))
        :param data_shape: shape of the real data (the tensor has no shape yet)
        :return: symbolic tensor for P, (Uxy, Uz, Wxy, Wz, beta, gamma, b)
        """

        exponent_, params_ = self._build_exponent(X_, data_shape)
        p_ = T.exp(exponent_).sum(axis=0)

        # apply logistic function to log p_ and add a bit of offset for numerical stability
        p2_ = p_ / (1 + p_) * (1 - 2 * 1e-8) + 1e-8
        return p2_, params_

    def __str__(self):
        return """
        Range degenerate Bernoulli process
        quadratic components: %i
        linear components: %i
        common components: %i
        """ % (self.quadratic_channels, self.linear_channels, self.common_channels)

    def __repr__(self):
        return self.__str__()
