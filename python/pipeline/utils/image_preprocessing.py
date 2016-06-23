import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.ndimage.filters import convolve1d

#

def histeq(x, bins=500, alpha=.9, beta=5):
    # get image histogram

    h, edges = np.histogram(x.ravel(), bins)
    cdf = h.cumsum().astype(float)  # cumulative distribution function
    cdf /= cdf[-1]  # normalize
    # use linear interpolation of cdf to find new pixel values
    # out = np.interp(x.ravel(), edges[:-1], cdf)
    target = stats.beta.ppf(cdf, alpha, beta)
    out = np.interp(x.ravel(), edges[:-1], target)
    out -= out.mean()
    return out.reshape(x.shape)


def local_standardize(X, kernelsize=(17, 17)):
    local_sq = X ** 2
    local_mean = np.asarray(X)
    for axis, ks in enumerate(kernelsize):
        w = np.hamming(ks)
        w /= w.sum()
        local_sq = convolve1d(local_sq, w, axis=axis, mode='reflect')
        local_mean = convolve1d(local_mean, w, axis=axis, mode='reflect')
    return  (X - local_mean)/ np.sqrt(local_sq - local_mean**2)


class App:
    def __init__(self, Ave_frame, Mon_frame, count, old_data_list, fig = None, ax = None):
        # The constructor.
        self.Ave_frame = Ave_frame
        self.Mon_frame = Mon_frame
        if fig is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.fig, self.ax = fig, ax

        self.current_frame = 'pre'
        self.redraw()
        self.fig.canvas.mpl_connect('button_press_event',self.on_press)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.count = count
        self.data_list = old_data_list

    def on_press(self, event):
        x, y = event.xdata, event.ydata
        plt.ion()
        if event.button == 1:
            if x >= 0 and x <= 250 and y >= 0 and y <= 250:
                self.ax.scatter(x, y, c='r')
                self.data_list.append([x, y, self.count])
                plt.pause(0.001)
                self.count += 1
            else:
                print('point out of bound')
        if event.button == 3:
            self.data_list = filter(lambda i: (i[0] < x - 2 or i[0] > x + 2) or (i[1] < y - 2 or i[1] > y + 2), self.data_list)
            temp = list(map(lambda x:[x[0], x[1]], self.data_list))
            x_array, y_array = zip(*temp)
            self.ax.clear()
            self.redraw()
            self.ax.scatter(list(x_array), list(y_array), c = 'r')
            plt.draw()

            for i in temp:
                i.append(temp.index(i) + 1)
            self.count = len(temp) + 1
            self.data_list = temp
        plt.ioff()

    def on_key(self, event):
        plt.ion()
        sns.set_style('white')
        if event.key == 'm':
            self.current_frame = 'monet'
            self.redraw()
        if event.key == 'p':
            self.current_frame = 'pre'
            self.redraw()
        plt.pause(0.001)
        plt.ioff()

    def redraw(self):
        plot_params = dict(cmap=plt.cm.get_cmap('bwr'))
        if self.current_frame == 'pre':
            self.ax.imshow(self.Ave_frame, **plot_params)
        else:
            self.ax.imshow(np.log(self.Mon_frame), **plot_params)
        plt.title(self.current_frame)
        plt.gca().set_xlim([0, 250])
        plt.gca().set_ylim([0, 250])