import os
# disable DLC GUI
os.environ["DLClight"] = "True"

from deeplabcut.utils import plotting
from deeplabcut.utils import video_processor
from deeplabcut.utils import auxiliaryfunctions
import deeplabcut as dlc

from IPython import display
import pylab as pl

import math
import yaml
import ruamel.yaml
import cv2
from pathlib import Path
import numpy as np
import pandas as pd
import imageio
import time
import shutil
import matplotlib.pyplot as plt

def key_dict_generater(case):
    case_key = {'animal_id': None, 'session': None, 'scan_idx': None}
    for ind, key in enumerate(case_key.keys()):
        case_key[key] = int(case.split('_')[ind])

    return case_key


def make_circumcircle(a, b, c):
    # Mathematical algorithm from Wikipedia: Circumscribed circle
    ox = (min(a[0], b[0], c[0]) + max(a[0], b[0], c[0])) / 2.0
    oy = (min(a[1], b[1], c[1]) + max(a[1], b[1], c[1])) / 2.0
    ax = a[0] - ox
    ay = a[1] - oy
    bx = b[0] - ox
    by = b[1] - oy
    cx = c[0] - ox
    cy = c[1] - oy
    d = (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) * 2.0
    if d == 0.0:
        return None
    x = ox + ((ax*ax + ay*ay) * (by - cy) + (bx*bx + by*by)
              * (cy - ay) + (cx*cx + cy*cy) * (ay - by)) / d
    y = oy + ((ax*ax + ay*ay) * (cx - bx) + (bx*bx + by*by)
              * (ax - cx) + (cx*cx + cy*cy) * (bx - ax)) / d
    ra = math.hypot(x - a[0], y - a[1])
    rb = math.hypot(x - b[0], y - b[1])
    rc = math.hypot(x - c[0], y - c[1])
    return (x, y, max(ra, rb, rc))


_MULTIPLICATIVE_EPSILON = 1 + 1e-14


def is_in_circle(c, p):
    return c is not None and math.hypot(p[0] - c[0], p[1] - c[1]) <= c[2] * _MULTIPLICATIVE_EPSILON


def smallest_enclosing_circle_naive(points):
    # Degenerate cases
    if len(points) == 0:
        return None
    elif len(points) == 1:
        return (points[0][0], points[0][1], 0)

    result = None
    # Try all unique triples
    for i in range(len(points)):
        p = points[i]
        for j in range(i + 1, len(points)):
            q = points[j]
            for k in range(j + 1, len(points)):
                r = points[k]
                c = make_circumcircle(p, q, r)
                if c is not None and (result is None or c[2] < result[2]) and \
                        all(is_in_circle(c, s) for s in points):
                    result = c

    if result is None:
        raise AssertionError()
    return result


def online_median_filter(x, kernel_size=3):

    assert kernel_size%2 == 1, "kernel size must be odd number!"

    interval = kernel_size//2

    online_medfilt = x[0:interval].tolist()
    for i in range(interval, len(x)-interval):
        online_medfilt.append(np.median(x[i-interval:i+interval+1]))
        
    online_medfilt += x[-interval:].tolist()

    return np.array(online_medfilt)


class DeeplabcutPlotBodyparts():

    def __init__(self, config, bodyparts='all', cropped=False):
        """
        Input:
            config: dictionary
                A dictionary that contains animal id, session, scan idx, and a path to config
            bodyparts: list
                A list that contains bodyparts to plot. Each bodypart is in a string format. If none provided,
                then by default it plots ALL existing bodyplots in config.yaml file.
            cropped: boolean
                whether to crop the video or not. Default False
            filtering (dict):

        """

        self.config = config

        if isinstance(bodyparts, list):
            self.bodyparts = bodyparts
        else:
            self.bodyparts = self.config['bodyparts']

        self.cropped = cropped
        self.cropped_coords = config['cropped_coords']

        self.shuffle = self.config['shuffle']
        self.trainingsetindex = self.config['trainingsetindex']

        self.project_path = self.config['project_path']
        self.orig_video_path = self.config['orig_video_path']
        self.base_dir = os.path.dirname(self.orig_video_path)
        self.compressed_cropped_dir_path = os.path.join(
            os.path.dirname(self.orig_video_path), 'compressed_cropped')
        self._case = os.path.basename(config['orig_video_path']).split('.')[0]
        self.clip = video_processor.VideoProcessorCV(
            fname=self.orig_video_path)

        self._trainFraction = self.config['TrainingFraction'][self.trainingsetindex]
        self._DLCscorer = auxiliaryfunctions.GetScorerName(
            self.config, self.shuffle, self._trainFraction)

        self.label_path = os.path.join(
            self.compressed_cropped_dir_path, self._case + '_compressed_cropped' + self._DLCscorer + '.h5')

        self.df_label = pd.read_hdf(self.label_path)

        self.df_bodyparts = self.df_label[self._DLCscorer][self.bodyparts]

        self.df_bodyparts_likelihood = self.df_bodyparts.iloc[:, self.df_bodyparts.columns.get_level_values(
            1) == 'likelihood']

        self.df_bodyparts_x = self.df_bodyparts.iloc[:,
                                                     self.df_bodyparts.columns.get_level_values(1) == 'x']
        self.df_bodyparts_y = self.df_bodyparts.iloc[:,
                                                     self.df_bodyparts.columns.get_level_values(1) == 'y']

        # in mm. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3310398/#R13
        self._pupil_diameter = 3.0 

        # obtain median left to right eyelid distance
        right = self.df_bodyparts['eyelid_right'].values[:, :2]
        left = self.df_bodyparts['eyelid_left'].values[:, :2]

        self.median_left_right = np.median(
            np.sqrt(np.einsum('ij,ij->i', left-right, left-right)))

        # obtain pixel to diameter ratio
        self._pixel_to_diameter_ratio = self.median_left_right/self._pupil_diameter

        if not self.cropped:

            self.nx = self.clip.width()
            self.ny = self.clip.height()
            self.df_bodyparts_x = self.df_bodyparts.iloc[:,
                                                         self.df_bodyparts.columns.get_level_values(1) == 'x'] + self.cropped_coords[0]
            self.df_bodyparts_y = self.df_bodyparts.iloc[:,
                                                         self.df_bodyparts.columns.get_level_values(1) == 'y'] + self.cropped_coords[2]

        else:

            if self.cropped_coords is None:
                raise ValueError(
                    "cropped_coords are not provided! Must be in list with 4 elmnts long!")

            if len(self.cropped_coords) != 4:
                raise ValueError(
                    "Only provided {} coordinates! U need 4!".format(len(self.cropped_coords)))

            # self.df_bodyparts_x = self.df_bodyparts.iloc[:,
            #                                             self.df_bodyparts.columns.get_level_values(1) == 'x'] - self.cropped_coords[0]
            # self.df_bodyparts_y = self.df_bodyparts.iloc[:,
            #                                             self.df_bodyparts.columns.get_level_values(1) == 'y'] - self.cropped_coords[2]

            self.nx = self.clip.width() - self.cropped_coords[0]
            self.ny = self.clip.height() - self.cropped_coords[2]

        # plotting properties
        self._dotsize = 7
        self._line_thickness = 1
        self._pcutoff = self.config['pcutoff']
        self._colormap = self.config['colormap']
        self._label_colors = plotting.get_cmap(
            len(self.bodyparts), name=self._colormap)
        self._alphavalue = self.config['alphavalue']
        self._fig_size = [12, 8]
        self._dpi = 100
        self._fontsize = 30

        self.tf_likelihood_array = self.df_bodyparts_likelihood.values > self._pcutoff

    @property
    def dotsize(self):
        return self._dotsize

    @dotsize.setter
    def dotsize(self, value):
        self._dotsize = value

    @property
    def line_thickness(self):
        return self._line_thickness

    @line_thickness.setter
    def line_thickness(self, value):
        if isinstance(value, int):
            self._line_thickness = value

        else:
            raise TypeError("line thickness must be integer")

    @property
    def pcutoff(self):
        return self._pcutoff

    @pcutoff.setter
    def pcutoff(self, value):
        self._pcutoff = value
        self.tf_likelihood_array = self.df_bodyparts_likelihood.values > self._pcutoff

    @property
    def colormap(self):
        return self._colormap

    @colormap.setter
    def colormap(self, value):
        if isinstance(value, str):
            self._colormap = value
            self._label_colors = plotting.get_cmap(
                len(self.bodyparts), name=self._colormap)
        else:
            raise TypeError("colormap must be in string format")

    @property
    def alphavalue(self):
        return self._alphavalue

    @alphavalue.setter
    def alphavalue(self, value):
        self._alphavalue = value

    @property
    def fig_size(self):
        return self._fig_size

    @fig_size.setter
    def fig_size(self, value):
        if isinstance(value, list):
            self._fig_size = value
        else:
            raise TypeError("fig_size must be in a list format")

    @property
    def dpi(self):
        return self._dpi

    @dpi.setter
    def dpi(self, value):
        self._dpi = value

    @property
    def fontsize(self):
        return self._fontsize

    @fontsize.setter
    def fontsize(self, value):
        self._fontsize = value

    @property
    def pupil_diameter(self):
        return self._pupil_diameter

    @pupil_diameter.setter
    def pupil_diameter(self, value):
        self._pupil_diameter = value

    @property
    def pixel_to_diameter_ratio(self):
        return self.median_left_right/self._pupil_diameter
    
    def coords_pcutoff(self, frame_num):
        """
        Given a frame number, return bpindex, x & y coordinates that meet pcutoff criteria
        Input:
            frame_num: int
                A desired frame number
        Output:
            bpindex: list
                A list of integers that match with bodypart. For instance, if the bodypart is ['A','B','C']
                and only 'A' and 'C'qualifies the pcutoff, then bpindex = [0,2]
            x_coords: pandas series
                A pandas series that contains coordinates whose values meet pcutoff criteria
            y_coords: pandas series
                A pandas series that contains coordinates whose values meet pcutoff criteria
        """
        frame_num_tf = self.tf_likelihood_array[frame_num, :]
        bpindex = [i for i, x in enumerate(frame_num_tf) if x]

        df_x_coords = self.df_bodyparts_x.loc[frame_num, :][bpindex]
        df_y_coords = self.df_bodyparts_y.loc[frame_num, :][bpindex]

        return bpindex, df_x_coords, df_y_coords

    def configure_plot(self):
        fig = plt.figure(frameon=False, figsize=self.fig_size, dpi=self.dpi)
        ax = fig.add_subplot(1, 1, 1)
        plt.subplots_adjust(left=0, bottom=0, right=1,
                            top=1, wspace=0, hspace=0)
        plt.xlim(0, self.nx)
        plt.ylim(0, self.ny)

        plt.gca().invert_yaxis()

        sm = plt.cm.ScalarMappable(cmap=self._label_colors, norm=plt.Normalize(
            vmin=-0.5, vmax=len(self.bodyparts)-0.5))
        sm._A = []
        cbar = plt.colorbar(sm, ticks=range(len(self.bodyparts)))
        cbar.set_ticklabels(self.bodyparts)
        cbar.ax.tick_params(labelsize=18)

        return fig, ax

    def plot_core(self, ax, frame_num):
        # it's given in 3 channels but every channel is the same i.e. grayscale
        image = self.clip._read_specific_frame(frame_num)

        if self.cropped:

            x1 = self.cropped_coords[0]
            x2 = self.cropped_coords[1]
            y1 = self.cropped_coords[2]
            y2 = self.cropped_coords[3]

            image = image[y1:y2, x1:x2]

        ax_frame = ax.imshow(image, cmap='gray')

        # plot bodyparts above the pcutoff
        bpindex, df_x_coords, df_y_coords = self.coords_pcutoff(frame_num)
        ax_scatter = ax.scatter(df_x_coords.values, df_y_coords.values, s=self.dotsize**2,
                                color=self._label_colors(bpindex), alpha=self.alphavalue)

        return {'ax_frame': ax_frame, 'ax_scatter': ax_scatter}

    def plot_one_frame(self, frame_num, ax=None, save_fig=False):
        # TODO refactor to reflect the sytle of plot_fitted_frame
        if ax is None:
            fig, ax = self.configure_plot()

        ax_dict = self.plot_core(ax, frame_num)

        ax.axis('off')
        ax.set_title('frame num: ' + str(frame_num), fontsize=self.fontsize)
        plt.tight_layout()

        if save_fig:
            plt.savefig(os.path.join(
                self.video_path.split('.')[0] + '_frame_' + str(frame_num) + '.png'))

        # return ax_dict
        if ax is not None:
            return ax

    def plot_multi_frames(self, start, end, save_gif=False):

        fig, ax = self.configure_plot()

        plt_list = []

        for frame_num in range(start, end):
            ax_dict = self.plot_core(ax, frame_num)

            plt.axis('off')
            plt.title('frame num: ' + str(frame_num), fontsize=self.fontsize)
            plt.tight_layout()

            fig.canvas.draw()

            data = np.fromstring(fig.canvas.tostring_rgb(),
                                 dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt_list.append(data)

            display.clear_output(wait=True)
            display.display(pl.gcf())
            time.sleep(0.5)

            plt.cla()

        if save_gif:
            gif_path = self.video_path.split('.')[0] + '_'\
                + str(start) + '_' + str(end) + '.gif'

            imageio.mimsave(gif_path, plt_list, fps=1)

        plt.close('all')

    def make_movie(self, start, end):

        import matplotlib.animation as animation

        # initlize with start frame
        fig, ax = self.configure_plot()

        self.plot_one_frame(frame_num=start, ax=ax)

        def update_frame(frame_num):

            # clear out the axis
            plt.cla()

            self.plot_one_frame(frame_num=frame_num, ax=ax)

        ani = animation.FuncAnimation(fig, update_frame, range(
            start+1, end))  # , interval=int(1/self.clip.FPS)
        # ani = animation.FuncAnimation(fig, self.plot_fitted_frame, 10)
        writer = animation.writers['ffmpeg'](fps=self.clip.FPS)

        # dpi=self.dpi, fps=self.clip.FPS
        if self.cropped:
            crop_flag = 'cropped'
        else:
            crop_flag = 'orig'
        save_video_path = os.path.join(self.base_dir,
                                       '{}_{}_{}_labeled.avi'.format(crop_flag, start, end))
        ani.save(save_video_path, writer=writer, dpi=self.dpi)

        return ani


class DeeplabcutPupilFitting(DeeplabcutPlotBodyparts):
    def __init__(self, config, bodyparts='all', cropped=False, filtering=None):
        """
        Input:
            config: dictionary
                A dictionary that contains animal id, session, scan idx, and a path to config
            bodyparts: list
                A list that contains bodyparts to plot. Each bodypart is in a string format. If none provided,
                then by default it plots ALL existing bodyplots in config.yaml file.

        """
        super().__init__(config, bodyparts=bodyparts, cropped=cropped)

        self.complete_eyelid_graph = {'eyelid_top': 'eyelid_top_right',
                                      'eyelid_top_right': 'eyelid_right',
                                      'eyelid_right': 'eyelid_right_bottom',
                                      'eyelid_right_bottom': 'eyelid_bottom',
                                      'eyelid_bottom': 'eyelid_bottom_left',
                                      'eyelid_bottom_left': 'eyelid_left',
                                      'eyelid_left': 'eyelid_left_top',
                                      'eyelid_left_top': 'eyelid_top'}

        self._circle_threshold_num = 3
        self._ellipse_threshold_num = 6
        self._circle_color = (0, 255, 0)
        self._ellipse_color = (0, 0, 255)

    @property
    def circle_threshold_num(self):
        return self._circle_threshold_num

    @circle_threshold_num.setter
    def circle_threshold_num(self, value):
        if value > len(self.complete_eyelid_graph.keys()):
            raise ValueError("value must be equal to or less than {}!".format(
                len(self.complete_eyelid_graph.keys())))
        else:
            self._circle_threshold_num = value

    @property
    def ellipse_threshold_num(self):
        return self._ellipse_threshold_num

    @ellipse_threshold_num.setter
    def ellipse_threshold_num(self, value):
        if value > len(self.complete_eyelid_graph.keys()):
            raise ValueError("value must be equal to or less than {}!".format(
                len(self.complete_eyelid_graph.keys())))

        # 5 is the minimum number needed for ellipse fitting
        elif value < 5:
            raise ValueError("value must be equal to or more than 5!")
        else:
            self._ellipse_threshold_num = value

    @property
    def circle_color(self):
        return self._circle_color

    @circle_color.setter
    def circle_color(self, value):
        self._circle_color = value

    @property
    def ellipse_color(self):
        return self._ellipse_color

    @ellipse_color.setter
    def ellipse_color(self, value):
        self._ellipse_color = value

    def coords_pcutoff(self, frame_num):
        """
        Given a frame number, return bpindex, x & y coordinates that meet pcutoff criteria
        Input:
            frame_num: int
                A desired frame number
        Output:
            bpindex: list
                A list of integers that match with bodypart. For instance, if the bodypart is ['A','B','C']
                and only 'A' and 'C'qualifies the pcutoff, then bpindex = [0,2]
            x_coords: pandas series
                A pandas series that contains coordinates whose values meet pcutoff criteria
            y_coords: pandas series
                A pandas series that contains coordinates whose values meet pcutoff criteria
        """
        frame_num_tf = self.tf_likelihood_array[frame_num, :]
        bpindex = [i for i, x in enumerate(frame_num_tf) if x]

        df_x_coords = self.df_bodyparts_x.loc[frame_num, :][bpindex]
        df_y_coords = self.df_bodyparts_y.loc[frame_num, :][bpindex]

        return bpindex, df_x_coords, df_y_coords

    def connect_eyelids(self, frame_num, frame):
        """
        connect eyelid labels with a straight line. If a label is missing, do not connect and skip to the next label.
        Input:
            frame_num: int
                A desired frame number
            frame: numpy array
                A frame to be fitted
        Output:
            A dictionary containing the fitted frame and its corresponding binary mask.
            For each key in dictionary:
                frame:
                    A numpy array frame with eyelids connected
                mask:
                    A binary numpy array
        """
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        _, df_x_coords, df_y_coords = self.coords_pcutoff(frame_num)
        eyelid_labels = [label for label in list(
            df_x_coords.index.get_level_values(0)) if 'eyelid' in label]

        for eyelid in eyelid_labels:
            next_bp = self.complete_eyelid_graph[eyelid]

            if next_bp not in eyelid_labels:
                continue

            coord_0 = tuple(
                map(int, map(round, [df_x_coords[eyelid].values[0], df_y_coords[eyelid].values[0]])))
            coord_1 = tuple(
                map(int, map(round, [df_x_coords[next_bp].values[0], df_y_coords[next_bp].values[0]])))
            # opencv has some issues with dealing with np objects. Cast it manually again
            frame = cv2.line(
                np.array(frame), coord_0, coord_1, color=(255, 0, 0), thickness=self.line_thickness)
            mask = cv2.line(
                mask, coord_0, coord_1, color=(255), thickness=self.line_thickness)

        # fill out the mask with 1s OUTSIDE of the mask, then invert 0 and 1
        # for cv2.floodFill, need a mask that is 2 pixels bigger than the input image
        new_mask = np.zeros((mask.shape[0]+2, mask.shape[1]+2), dtype=np.uint8)
        cv2.floodFill(mask, new_mask, seedPoint=(0, 0), newVal=124)

        final_mask = np.logical_not(new_mask).astype(int)[1:-1, 1:-1]

        # ax.imshow(mask)
        return {'frame': frame,
                'mask': final_mask,
                'eyelid_labels_num': len(eyelid_labels)}

    def fit_circle_to_pupil(self, frame_num, frame):
        """
        Fit a circle to the pupil if it meets the circle_threshold value
        Input:
            frame_num: int
                A desired frame number
            frame: numpy array
                A frame to be fitted 3D

        Output: dictionary
            A dictionary with the fitted frame, center and radius of the fitted circle. If fitting did
            not occur, return the original frame with center and raidus as None.
            For each key in dictionary:
                frame: numpy array
                    a numpy array of the frame with pupil circle
                center: tuple
                    coordinates of the center of the fitted circle. In tuple format
                radius: float
                    radius of the fitted circle in int format
                mask: numpy array
                    a binary mask for the fitted circle area
                pupil_labels_num: int
                    number of pupil labels used for fitting
        """

        mask = np.zeros(frame.shape, dtype=np.uint8)

        _, df_x_coords, df_y_coords = self.coords_pcutoff(frame_num)

        pupil_labels = [label for label in list(
            df_x_coords.index.get_level_values(0)) if 'pupil' in label]

        if len(pupil_labels) < self.circle_threshold_num:
            # print('Frame number: {} has only 2 or less pupil label. Skip fitting!'.format(
            #     frame_num))
            center = None
            radius = None
            final_mask = mask[:, :, 0]

        else:
            pupil_x = df_x_coords.loc[pupil_labels].values
            pupil_y = df_y_coords.loc[pupil_labels].values

            pupil_coords = list(zip(pupil_x, pupil_y))

            x, y, radius = smallest_enclosing_circle_naive(pupil_coords)

            center = (x, y)

            # opencv has some issues with dealing with np objects. Cast it manually again
            frame = cv2.circle(img=np.array(frame), center=(int(round(x)), int(round(y))),
                               radius=int(round(radius)), color=self.circle_color, thickness=self.line_thickness)

            mask = cv2.circle(img=mask, center=(int(round(x)), int(round(y))),
                              radius=int(round(radius)), color=self.circle_color, thickness=self.line_thickness)

            # fill out the mask with 1s OUTSIDE of the mask, then invert 0 and 1
            # for cv2.floodFill, need a mask that is 2 pixels bigger than the input image
            new_mask = np.zeros(
                (mask.shape[0]+2, mask.shape[1]+2), dtype=np.uint8)
            cv2.floodFill(mask, new_mask, seedPoint=(0, 0), newVal=1)
            final_mask = np.logical_not(new_mask).astype(int)[1:-1, 1:-1]

        return {'frame': frame,
                'center': center,
                'radius': radius,
                'mask': final_mask,
                'pupil_labels_num': len(pupil_labels)}

    def fit_ellipse_to_pupil(self, frame_num, frame):
        """
        Fit an ellipse to pupil iff there exist more than 6 labels. If less than 6, return None
        Input:
            frame_num: int
                A desired frame number
            frame: numpy array
                A frame to be fitted in 3D
        Output: dictionary
            A dictionary with the fitted frame, center and radius of the fitted ellipse. If fitting did
            not occur, return None.
            For each key in dictionary:
                frame: numpy array
                    a numpy array of the frame with pupil ellipse
                center: tuple 
                    coordinates of the center of the fitted ellipse in tuple of floats
                mask: numpy array
                    a binary mask for the fitted ellipse area
                major_radius: float
                    major radius of the fitted ellipse
                minor_radius: float
                    minor radius of the fitted ellipse
                rotation_angle: float
                    angle from degree 0 to major_radius                
                pupil_labels_num: int
                    number of pupil labels used for fitting
        """

        mask = np.zeros(frame.shape, dtype=np.uint8)

        _, df_x_coords, df_y_coords = self.coords_pcutoff(frame_num)

        pupil_labels = [label for label in list(
            df_x_coords.index.get_level_values(0)) if 'pupil' in label]

        if len(pupil_labels) < self.ellipse_threshold_num:
            # print('Frame number: {} has only 2 or less pupil label. Skip fitting!'.format(
            #     frame_num))
            center = None
            major_radius = None
            minor_radius = None
            rotation_angle = None
            final_mask = mask[:, :, 0]

        else:
            pupil_x = df_x_coords.loc[pupil_labels].values.round().astype(
                np.int32)
            pupil_y = df_y_coords.loc[pupil_labels].values.round().astype(
                np.int32)

            pupil_coords = np.array(
                list(zip(pupil_x, pupil_y))).reshape((-1, 1, 2))

            # https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#fitellipse
            # Python: cv.FitEllipse2(points) → Box2D
            # https://docs.opencv.org/3.4.5/db/dd6/classcv_1_1RotatedRect.html
            rotated_rect = cv2.fitEllipse(pupil_coords)

            # https://docs.opencv.org/3.0-beta/modules/imgproc/doc/drawing_functions.html#ellipse
            # cv2.ellipse(img, box, color[, thickness[, lineType]]) → img
            frame = cv2.ellipse(np.array(
                frame), rotated_rect, color=self.ellipse_color, thickness=self.line_thickness)
            mask = cv2.ellipse(np.array(
                mask), rotated_rect, color=self.ellipse_color, thickness=self.line_thickness)

            # fill out the mask with 1s OUTSIDE of the mask, then invert 0 and 1
            # for cv2.floodFill, need a mask that is 2 pixels bigger than the input image
            new_mask = np.zeros(
                (mask.shape[0]+2, mask.shape[1]+2), dtype=np.uint8)
            cv2.floodFill(mask, new_mask, seedPoint=(0, 0), newVal=1)
            final_mask = np.logical_not(new_mask).astype(int)[1:-1, 1:-1]

            center = rotated_rect[0]
            major_radius = rotated_rect[1][1]/2.0
            minor_radius = rotated_rect[1][0]/2.0
            rotation_angle = rotated_rect[2]

        return {'frame': frame,
                'center': center,
                'mask': final_mask,
                'major_radius': major_radius,
                'minor_radius': minor_radius,
                'rotation_angle': rotation_angle,
                'pupil_labels_num': len(pupil_labels)}

    def detect_visible_pupil_area(self, eyelid_connect_dict, fit_dict, fitting_method=None):
        """
        Given a frame, find a visible part of the pupil by finding the intersection of pupil and eyelid masks
        If pupil mask does not exist(i.e. label < 3), return None

        Input:
            eyelid_connect_dict: dicionary
                An output dictionary from method 'connect_eyelids'
            fit_dict: dictionary
                An output dictionary from either of the method 'fit_circle_to_pupil' or 'fit_ellipse_to_pupil'
            fitting_method: string
                A string indicates whether the fitted method was an 'ellipse' or a 'circle'. 

        Output:
            A dictionary that contains the following:
                mask: numpy array
                    A 3D mask that depicts visible area of pupil. 
                    If no visible area provided, then it is an np.zeros
                visible_portion: signed float
                    if visible area exists, then value ranges from 0.0 to 1.0
                    if equal to -1.0, not all eyelid labels exist, hence cannot find the visible area
                    if equal to -2.0, number of pupil labels do not meet the threshold of the fitting method, hence no fitting performed
                    if equal to -3.0, not all eyelid labels exist AND not enough pupil labels to meet the threshold of the fitting method
        """

        color_mask = np.zeros(
            shape=[*eyelid_connect_dict['mask'].shape, 3], dtype=np.uint8)

        if fitting_method == 'circle':
            threshold = self.circle_threshold_num

        elif fitting_method == 'ellipse':
            threshold = self.ellipse_threshold_num

        else:
            raise ValueError(
                'fitting_method must be provided! It must be either a "circle" or an "ellipse"')

        if fit_dict['pupil_labels_num'] >= threshold and eyelid_connect_dict['eyelid_labels_num'] == 8:

            visible_mask = np.logical_and(
                fit_dict['mask'], eyelid_connect_dict['mask']).astype(int)

            # 126,0,255 for the color
            color_mask[visible_mask == 1, 0] = 126
            color_mask[visible_mask == 1, 2] = 255

            visible_portion = visible_mask.sum() / fit_dict['mask'].sum()

        elif fit_dict['pupil_labels_num'] >= threshold and eyelid_connect_dict['eyelid_labels_num'] != 8:
            visible_portion = -1.0

        elif fit_dict['pupil_labels_num'] < threshold and eyelid_connect_dict['eyelid_labels_num'] == 8:
            visible_portion = -2.0

        elif fit_dict['pupil_labels_num'] < threshold and eyelid_connect_dict['eyelid_labels_num'] != 8:
            visible_portion = -3.0

        return dict(mask=color_mask, visible_portion=visible_portion)

    def fitted_core(self, frame_num):
        """
        Input:
            fig: figure object
                Figure object created by configure_plot method
            ax: axis object
                Axis object created by configure_plot method
            frame_num: int
                frame number of the video to be analyzed
            fitting_method: string
                It must be either 'circle' or 'ellipse'. If not provided, raising an error
        Output:
            A dictionary
                'ax_frame': A fitted axis object
                'ax_scatter': A scatter axis object that shows where labels are
                'ax_mask':
        """
        # it's given in 3 channels but every channel is the same i.e. grayscale

        image = self.clip._read_specific_frame(frame_num)

        if self.cropped:

            x1 = self.cropped_coords[0]
            x2 = self.cropped_coords[1]
            y1 = self.cropped_coords[2]
            y2 = self.cropped_coords[3]

            image = image[y1:y2, x1:x2]

        eyelid_connected = self.connect_eyelids(
            frame_num=frame_num, frame=image)

        circle_fit = self.fit_circle_to_pupil(
            frame_num=frame_num, frame=eyelid_connected['frame'])

        ellipse_fit = self.fit_ellipse_to_pupil(
            frame_num=frame_num, frame=eyelid_connected['frame'])

        circle_visible = self.detect_visible_pupil_area(eyelid_connect_dict=eyelid_connected,
                                                        fit_dict=circle_fit,
                                                        fitting_method='circle')

        ellipse_visible = self.detect_visible_pupil_area(eyelid_connect_dict=eyelid_connected,
                                                         fit_dict=ellipse_fit,
                                                         fitting_method='ellipse')

        return {'circle_fit': circle_fit,
                'ellipse_fit': ellipse_fit,
                'circle_visible': circle_visible,
                'ellipse_visible': ellipse_visible}

    def plot_fitted_frame(self, frame_num, ax=None, fitting_method='circle', save_fig=False):

        if ax is None:
            fig, ax = self.configure_plot()

        # plot bodyparts above the pcutoff
        bpindex, x_coords, y_coords = self.coords_pcutoff(frame_num)
        ax_scatter = ax.scatter(x_coords.values, y_coords.values, s=self.dotsize**2,
                                color=self._label_colors(bpindex), alpha=self.alphavalue)

        fitted_core_dict = self.fitted_core(frame_num)

        if fitting_method == 'circle':

            circle_frame = ax.imshow(fitted_core_dict['circle_fit']['frame'])
            circle_mask = ax.imshow(
                fitted_core_dict['circle_visible']['mask'], alpha=0.2)

        elif fitting_method == 'ellipse':

            ellipse_frame = ax.imshow(fitted_core_dict['ellipse_fit']['frame'])
            ellipse_mask = ax.imshow(
                fitted_core_dict['ellipse_visible']['mask'], alpha=0.2)

        else:
            raise ValueError(
                'fitting method must be either an ellipse or a circle!')

        ax.set_title('frame num: ' + str(frame_num), fontsize=self.fontsize)

        ax.axis('off')
        plt.tight_layout()

        if save_fig:
            plt.savefig(os.path.join(
                self.compressed_cropped_dir_path, 'fitted_frame_' + str(frame_num) + '.png'))

        if ax is not None:
            return ax

    def plot_fitted_multi_frames(self, start, end, fitting_method='circle', save_gif=False):

        fig, ax = self.configure_plot()

        plt_list = []

        for frame_num in range(start, end):

            self.plot_fitted_frame(frame_num=frame_num,
                                   ax=ax, fitting_method=fitting_method)

            fig.canvas.draw()

            data = np.fromstring(fig.canvas.tostring_rgb(),
                                 dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt_list.append(data)

            display.clear_output(wait=True)
            display.display(pl.gcf())
            time.sleep(0.5)

            plt.cla()

        if save_gif:
            gif_path = self.video_path.split('.')[0] + '_fitted_' + \
                str(start) + '_' + str(end) + '.gif'
            imageio.mimsave(gif_path, plt_list, fps=1)

        plt.close('all')

    def make_movie(self, start, end, fitting_method='circle'):

        import matplotlib.animation as animation

        # initlize with start frame
        fig, ax = self.configure_plot()

        self.plot_fitted_frame(frame_num=start, ax=ax,
                               fitting_method=fitting_method)

        def update_frame(frame_num):

            # clear out the axis
            plt.cla()

            self.plot_fitted_frame(frame_num=frame_num,
                                   ax=ax, fitting_method=fitting_method)

        ani = animation.FuncAnimation(fig, update_frame, range(
            start+1, end))  # , interval=int(1/self.clip.FPS)
        # ani = animation.FuncAnimation(fig, self.plot_fitted_frame, 10)
        writer = animation.writers['ffmpeg'](fps=self.clip.FPS)

        if self.cropped:
            crop_flag = 'cropped'
        else:
            crop_flag = 'orig'
        save_video_path = os.path.join(self.base_dir,
                                       '{}_{}_{}_labeled.avi'.format(crop_flag, start, end))

        ani.save(save_video_path, writer=writer, dpi=self.dpi)

        return ani


def make_short_video(tracking_dir):
    """
    Extract 5 seconds long video starting from the middle of the original video.

    Input:
        tracking_dir: string
            String that specifies the full path of tracking directory
    Return:
        short_vid_path: string
            String that specifies the full path of short video
        original_width: int
            width of the original video
        original_height: int
            height of the original video
        mid_frame_num: int
            middle frame number of the original video

    """
    from subprocess import Popen, PIPE

    suffix = '_short.avi'

    case = os.path.basename(os.path.normpath(
        tracking_dir)).split('_tracking')[0]

    input_video_path = os.path.join(tracking_dir, case + '.avi')

    short_vid_path = os.path.join(tracking_dir, 'short', case + suffix)

    cap = cv2.VideoCapture(input_video_path)

    original_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    original_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    fps = cap.get(cv2.CAP_PROP_FPS)
    mid_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)/2)
    duration = int(mid_frame_num/fps)

    minutes, seconds = divmod(duration, 60)
    hours, minutes = divmod(minutes, 60)

    if os.path.exists(short_vid_path):
        # video already exists, even before processed. Hence delete it and recreate
        print('\nShort video already exists! This video might be corrupted, hence deleting and recreating!')

        # delete short video directory
        shutil.rmtree(os.path.dirname(short_vid_path))

        # recreate the short video directory
        os.mkdir(os.path.join(tracking_dir, 'short'))

    else:
        print('\nMaking a short video!')

        cmd = ['ffmpeg', '-i', input_video_path, '-ss',
               '{}:{}:{}'.format(hours, minutes, seconds), '-t', '5', '-c', 'copy', short_vid_path]

        # call ffmpeg to make a short video
        p = Popen(cmd, stdin=PIPE)
        # close ffmpeg
        p.wait()

        print('\nSuccessfully created a short video!')

    return short_vid_path, original_width, original_height, mid_frame_num


def predict_labels(vid_path, config, gputouse=0):
    """
    Predict labels on a given video

    Input:
        vid_path: string
            Path to video.
        config: dictionary
            a deeplabcut model configuration dictionary.
    """
    destfolder = os.path.dirname(vid_path)
    dlc.analyze_videos(config=config['config_path'], videos=[vid_path], videotype='avi', shuffle=config['shuffle'],
                       trainingsetindex=config['trainingsetindex'], gputouse=gputouse, save_as_csv=False, destfolder=destfolder)


def obtain_cropping_coords(short_h5_path, DLCscorer, config):
    """
    First, filter out by the pcutoff, then find values that are within 1 std from mean
    for each eyelid bodypart. Then, compare among the parts and find min,max values in x and y.

    The reason we use 1 std from mean is that dlc might have outliers in this short video.
    Hence we filter out these potential outliers

    Input:
        short_h5_path: string
            path to h5 file generated by deeplabcut on short video.
        DLCscorer: string
            scorer name used for deeplabcut. Can be obtained via auxiliaryfunctions.GetScorerName(config, shuffle, trainsetindex)
        config: dictionary
            a deeplabcut model configuration dictionary.
    """

    # there should be only 1 h5 file generated by dlc
    df_short = pd.read_hdf(short_h5_path)

    eyelid_cols = ['eyelid_top', 'eyelid_right',
                   'eyelid_left', 'eyelid_bottom']

    df_eyelid = df_short[DLCscorer][eyelid_cols]

    df_eyelid_likelihood = df_eyelid.iloc[:, df_eyelid.columns.get_level_values(
        1) == 'likelihood']
    df_eyelid_x = df_eyelid.iloc[:, df_eyelid.columns.get_level_values(
        1) == 'x']
    df_eyelid_y = df_eyelid.iloc[:, df_eyelid.columns.get_level_values(
        1) == 'y']

    df_eyelid_coord = dict(x=df_eyelid_x, y=df_eyelid_y)

    coords_dict = dict(xmin=[], xmax=[], ymin=[], ymax=[])

    for eyelid_label in eyelid_cols:

        for coord in ['x', 'y']:

            # only obtain if the labels are confident enough (i.e. > pcutoff)
            eyelid_coord_pcutoff = df_eyelid_coord[coord][(
                df_eyelid_likelihood.loc[:, eyelid_label].values > config['pcutoff'])][eyelid_label][coord].values

            # if the video is in bad quality, it is possible that none of the labels are above pcutoff results in an empty array.
            # If this happens, append original height and width
            if not eyelid_coord_pcutoff.size:
                if coord == 'x':
                    coords_dict[coord+'min'].append(0)
                    coords_dict[coord+'max'].append(config['original_width'])
                elif coord == 'y':
                    coords_dict[coord+'min'].append(0)
                    coords_dict[coord+'max'].append(config['original_height'])

                continue

            # only retain values within 1 std deviation from mean
            eyelid_coord_68 = eyelid_coord_pcutoff[(eyelid_coord_pcutoff < np.mean(eyelid_coord_pcutoff) + np.std(eyelid_coord_pcutoff)) *
                                                   (eyelid_coord_pcutoff > np.mean(
                                                    eyelid_coord_pcutoff) - np.std(eyelid_coord_pcutoff))]

            # sometimes, eyelid_coord_68 can return an empty array. If so, dont bother with 1st dev from mean 
            # but directly use eyelid_coord_pcutoff
            if len(eyelid_coord_68) == 0:
                coords_dict[coord+'min'].append(eyelid_coord_pcutoff.min())
                coords_dict[coord+'max'].append(eyelid_coord_pcutoff.max())
            else:
                coords_dict[coord+'min'].append(eyelid_coord_68.min())
                coords_dict[coord+'max'].append(eyelid_coord_68.max())

    cropped_coords = {}
    cropped_coords['cropped_x0'] = int(min(coords_dict['xmin']))
    cropped_coords['cropped_x1'] = int(max(coords_dict['xmax']))
    cropped_coords['cropped_y0'] = int(min(coords_dict['ymin']))
    cropped_coords['cropped_y1'] = int(max(coords_dict['ymax']))

    return cropped_coords


def add_pixels(cropped_coords, original_width, original_height, pixel_num):
    """
    Add addtional pixels around cropped_coords
    Input:
        cropped_coords: dictionary
            cropoping coordinates specifying left_top  and bottom_right coords
        original_width: int
            width of the original video
        original_height: int
            height of the original video
        pixel_num: int
            number of pixels to add around the cropped_coords
    Return:
        cropped_coords: dictionary
            updated cropoping coordinates specifying left_top  and bottom_right coords
    """

    if cropped_coords['cropped_x0'] - pixel_num < 0:
        cropped_coords['cropped_x0'] = 0
    else:
        cropped_coords['cropped_x0'] -= pixel_num

    if cropped_coords['cropped_x1'] + pixel_num > original_width:
        cropped_coords['cropped_x1'] = original_width
    else:
        cropped_coords['cropped_x1'] += pixel_num

    if cropped_coords['cropped_y0'] - pixel_num < 0:
        cropped_coords['cropped_y0'] = 0
    else:
        cropped_coords['cropped_y0'] -= pixel_num

    if cropped_coords['cropped_y1'] + pixel_num > original_height:
        cropped_coords['cropped_y1'] = original_height
    else:
        cropped_coords['cropped_y1'] += pixel_num

    return cropped_coords


def make_compressed_cropped_video(tracking_dir, cropped_coords):
    """
    Make a compressed and cropped video.

    Input:
        tracking_dir: string
            String that specifies the full path of tracking directory
        cropped_coords: dictionary
            cropoping coordinates specifying left_top  and bottom_right coords
    Return:
        None
    """
    from subprocess import Popen, PIPE

    suffix = '_compressed_cropped.avi'

    case = os.path.basename(os.path.normpath(
        tracking_dir)).split('_tracking')[0]

    input_video_path = os.path.join(tracking_dir, case + '.avi')

    cc_vid_path = os.path.join(
        tracking_dir, 'compressed_cropped', case + suffix)

    if os.path.exists(cc_vid_path):
        # video already exists, even before processed. Hence delete it and recreate
        print("\ncompressed and cropped video already exists! This video might be corrupted, hence deleting and recreating!")

        # delete short video directory
        shutil.rmtree(os.path.dirname(cc_vid_path))

        # recreate the short video directory
        os.mkdir(os.path.join(tracking_dir, 'compressed_cropped'))

    else:
        out_w = cropped_coords['cropped_x1'] - cropped_coords['cropped_x0']
        out_h = cropped_coords['cropped_y1'] - cropped_coords['cropped_y0']
        print('\nMaking a compressed and cropped video!')

        # crf: use value btw 17 and 28 (lower the number, higher the quality of the video)
        # intra: no compressing over time. only over space
        cmd = ['ffmpeg', '-i', '{}'.format(input_video_path), '-vcodec', 'libx264', '-crf', '17', '-intra', '-filter:v',
               "crop={}:{}:{}:{}".format(out_w, out_h, cropped_coords['cropped_x0'], cropped_coords['cropped_y0']), '{}'.format(cc_vid_path)]

        # call ffmpeg to make a short video
        p = Popen(cmd, stdin=PIPE)
        # close ffmpeg
        p.wait()
        print('\nSuccessfully created a compressed & cropped video!\n')

    return cc_vid_path


def filter_by_fitting_std(data, fitting_method, std_magnitude=5.5):
    """Filter out outliers based on std specified by user. The outlier indices are returned

    Args:
        data (numpy array): 
        if fitting_method is a circle
            0th column: center
            1st column: radius
            2nd column: visible portion
        if fitting method is an ellipse:
            0th column: center
            1st column: major_r
            2nd column: minor_r
            3rd column: visible portion

        fitting_method (str): A string specifying which fitting method used. Must be either a circle or an ellipse
        std_magnitude (float): A number that specifies how many std away from mean to be used as a cutoff.
            Default to 5.5 (emperically obtained value)

    Returns:
        rejected inds: rejected indices after filtered by std deviations. True is rejected.

    """
    if fitting_method.lower() == 'circle':
        # at minium we need center and radius info
        assert data.shape[1] >= 2

        # filter out circles
        center, radius = data[:, 0], data[:, 1].astype(np.float64)

        # only obtain real numbers, not nans.
        detectedFrames = ~np.isnan(radius)
        xy = np.full((len(radius), 2), np.nan)
        xy[detectedFrames, :] = np.vstack(center[detectedFrames])

        x = xy[:, 0]
        y = xy[:, 1]

        rejected_radius_ind = np.greater(abs(
            radius - np.nanmean(radius)), std_magnitude * np.nanstd(radius), where=~np.isnan(radius))
        rejected_x_ind = np.greater(
            abs(x - np.nanmean(x)), std_magnitude * np.nanstd(x), where=~np.isnan(x))
        rejected_y_ind = np.greater(
            abs(y - np.nanmean(y)), std_magnitude * np.nanstd(y), where=~np.isnan(y))

        rejected_ind = np.logical_or(np.logical_or(
            rejected_radius_ind, rejected_x_ind), rejected_y_ind)

    elif fitting_method.lower() == 'ellipse':
        # at minimum we need center, major_r, and minor_r info
        assert data.shape[1] >= 3

        # filter out ellipses
        center, major_r, minor_r = data[:, 0], data[:, 1].astype(
            np.float64), data[:, 2].astype(np.float64)

        detectedFrames = ~np.isnan(major_r)
        xy = np.full((len(major_r), 2), np.nan)
        xy[detectedFrames, :] = np.vstack(center[detectedFrames])

        x = xy[:, 0]
        y = xy[:, 1]

        rejected_major_r_ind = np.greater(abs(
            major_r - np.nanmean(major_r)), std_magnitude * np.nanstd(major_r), where=~np.isnan(major_r))
        rejected_minor_r_ind = np.greater(abs(
            minor_r - np.nanmean(minor_r)), std_magnitude * np.nanstd(minor_r), where=~np.isnan(minor_r))
        rejected_x_ind = np.greater(
            abs(x - np.nanmean(x)), std_magnitude * np.nanstd(x), where=~np.isnan(x))
        rejected_y_ind = np.greater(
            abs(y - np.nanmean(y)), std_magnitude * np.nanstd(y), where=~np.isnan(y))

        rejected_ind = np.logical_or(np.logical_or(np.logical_or(
            rejected_major_r_ind, rejected_minor_r_ind), rejected_x_ind), rejected_y_ind)

    return rejected_ind


