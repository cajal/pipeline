
import math
import os
import yaml
import cv2
from pathlib import Path
import numpy as np
import pandas as pd
import ruamel.yaml
import imageio
import time

from deeplabcut.utils import auxiliaryfunctions
from deeplabcut.utils.video_processor import VideoProcessorCV as vp
from deeplabcut.utils.plotting import get_cmap

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import pylab as pl
from IPython import display
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


# We just subclass Rectangle so that it can be called with an Axes
# instance, causing the rectangle to update its shape to match the
# bounds of the Axes


class UpdatingRect(Rectangle):
    def __call__(self, ax):
        self.set_bounds(*ax.viewLim.bounds)
        ax.figure.canvas.draw_idle()


class ZoomedDisplay():
    """
    Display the zoomed in area in the left panel.
    """

    def __init__(self, frame, height, width):
        self.frame = frame
        # in pixels
        self.xstart = 0
        self.xend = width
        self.ystart = 0
        self.yend = height

    def ax_update(self, ax):
        ax.set_autoscale_on(False)  # Otherwise, infinite loop

        # Get the range for the new area
        # viewLim.bounds give leftbottom and righttop coordinates
        xstart, yend, xdelta, ydelta = ax.viewLim.bounds
        xend = xstart + xdelta
        ystart = yend + ydelta

        self.xstart = round(xstart).astype(int)
        self.xend = round(xend).astype(int)
        self.ystart = round(ystart).astype(int)
        self.yend = round(yend).astype(int)

        # images are in row major order. user should double check if major order
        bounded_frame = self.frame[self.ystart:self.yend,
                                   self.xstart:self.xend]
        im = ax.images[-1]
        im.set_data(bounded_frame)
        # extent is data axes (left, right, bottom, top) for making image plots registered with data plots.
        # https://matplotlib.org/api/image_api.html#matplotlib.image.AxesImage.set_extent
        im.set_extent((xstart, xend, yend, ystart))
        ax.figure.canvas.draw_idle()


def update_config_crop_coords(config):
    """
    Given the list of videos, users can manually zoom in the area they want to crop and update the coordinates in config.yaml

    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.
    """
    config_file = Path(config).resolve()
    cfg = auxiliaryfunctions.read_config(config_file)

    video_sets = cfg['video_sets'].keys()
    for vindex, video_path in enumerate(video_sets):

        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(
                cv2.CAP_PROP_FRAME_WIDTH))

            print("video {}: {} has original dim in {} by {}".format(
                vindex, video_path, width, height))

            # putting the frame to read at the very middle of the video
            cap.set(cv2.CAP_PROP_POS_FRAMES, int((nframes-1)/2))
            res, frame = cap.read()

            display = ZoomedDisplay(frame=frame, height=height, width=width)

            fig1, (ax1, ax2) = plt.subplots(1, 2)

            ax1.imshow(frame)
            ax2.imshow(frame)

            rect = UpdatingRect([0, 0], 0, 0, facecolor='None',
                                edgecolor='red', linewidth=1.0)
            rect.set_bounds(*ax2.viewLim.bounds)
            ax1.add_patch(rect)

            # Connect for changing the view limits
            ax2.callbacks.connect('xlim_changed', rect)
            ax2.callbacks.connect('ylim_changed', rect)

            ax2.callbacks.connect('xlim_changed', display.ax_update)
            ax2.callbacks.connect('ylim_changed', display.ax_update)
            ax2.set_title("Zoom here")

            plt.show()

            new_width = display.xend - display.xstart
            new_height = display.yend - display.ystart

            print("your cropped coords are {} {} {} {} with dim of {} by {} \n".format(
                display.xstart, display.xend, display.ystart, display.yend, new_width, new_height))

            cfg['video_sets'][video_path] = {'crop': ', '.join(
                map(str, [display.xstart, display.xend, display.ystart, display.yend]))}

            cap.release()
            plt.close("all")

        else:
            print("Cannot open the video file: {} !".format(video_path))

    # Update the yaml config file
    auxiliaryfunctions.write_config(config, cfg)


def update_inference_cropping_config(cropping_config, video_path):
    """
    Given a video path, users can manually zoom in the area they want to crop and update cropping coordinates in config.yaml

    Parameters
    ----------
    cropping_config: string
        Full path of the inference_cropping.yaml file as a string.
    video_path : str
        Full path to the video

    Output:
        new cropping coordinates: list
            list contatining xstart, xend, ystart, and yend
    """
    config_file = Path(cropping_config).resolve()
    cfg = auxiliaryfunctions.read_config(config_file)

    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(
            cv2.CAP_PROP_FRAME_WIDTH))

        print("original dim in {} by {}".format(width, height))

        # putting the frame to read at the very middle of the video
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((nframes-1)/2))
        _, frame = cap.read()

        display = ZoomedDisplay(frame=frame, height=height, width=width)

        fig1, (ax1, ax2) = plt.subplots(1, 2)

        ax1.imshow(frame)
        ax2.imshow(frame)

        rect = UpdatingRect([0, 0], 0, 0, facecolor='None',
                            edgecolor='red', linewidth=1.0)
        rect.set_bounds(*ax2.viewLim.bounds)
        ax1.add_patch(rect)

        # Connect for changing the view limits
        ax2.callbacks.connect('xlim_changed', rect)
        ax2.callbacks.connect('ylim_changed', rect)

        ax2.callbacks.connect('xlim_changed', display.ax_update)
        ax2.callbacks.connect('ylim_changed', display.ax_update)
        ax2.set_title("Zoom here")

        plt.show()

        new_width = display.xend - display.xstart
        new_height = display.yend - display.ystart

        # Enforce that both width and height are even numbers for ffmpeg purpose
        if new_width % 2:
            display.xstart -= 1
            new_width = display.xend - display.xstart
        if new_height % 2:
            display.ystart -= 1
            new_height = display.yend - display.ystart

        print("your cropped coords are {} {} {} {} with dim of {} by {} \n".format(
            display.xstart, display.xend, display.ystart, display.yend, new_width, new_height))

        cfg[video_path]['x1'] = int(display.xstart)
        cfg[video_path]['x2'] = int(display.xend)
        cfg[video_path]['y1'] = int(display.ystart)
        cfg[video_path]['y2'] = int(display.yend)

        cap.release()
        plt.close("all")

        # Update the yaml config file
        yaml = ruamel.yaml.YAML()

        yaml.dump(cfg, config_file)

        return [int(display.xstart), int(display.xend), int(display.ystart), int(display.yend)]

    else:
        print("Cannot open the video file: {} !".format(video_path))


def crop_videos(cropping_config, case):
    from subprocess import Popen, PIPE

    config_file = Path(cropping_config).resolve()
    cfg = auxiliaryfunctions.read_config(config_file)

    suffix = '_compressed_cropped_beh.avi'

    video_dir = os.path.join(os.sep, 'work', 'videos')
    input_video_path = os.path.join(video_dir, case + '_beh.avi')

    output_vid = case + suffix
    out_vid_path = os.path.join(os.sep, 'work', 'videos', output_vid)

    if input_video_path not in dict(cfg).keys():
        raise KeyError(
            "Case: {} not in cropping_config.yaml. Did you add it yet?".format(case))

    if case+suffix in video_dir:
        print('case: {} is already cropped!'.format(case))
        return None

    xstart, xend, ystart, yend = list(dict(cfg)[input_video_path].values())[1:]
    out_w = xend - xstart
    out_h = yend - ystart

    cmd = ['ffmpeg', '-i', '{}'.format(input_video_path), '-vcodec', 'libx264', '-crf', '17', '-intra', '-filter:v',
           "crop={}:{}:{}:{}".format(out_w, out_h, xstart, ystart), '{}'.format(out_vid_path)]

    # call ffmpeg to crop and compress
    p = Popen(cmd, stdin=PIPE)
    # close ffmpeg
    p.wait()


class PlotBodyparts():

    def __init__(self, config, bodyparts='all'):
        """
        Input:
            config: dictionary
                A dictionary that contains animal id, session, scan idx, and a path to config
            bodyparts: list
                A list that contains bodyparts to plot. Each bodypart is in a string format. If none provided,
                then by default it plots ALL existing bodyplots in config.yaml file.

        """

        self.config = config

        if isinstance(bodyparts, list):
            self.bodyparts = bodyparts
        else:
            self.bodyparts = self.config['bodyparts']
        self.shuffle = self.config['shuffle']
        self.trainingsetindex = self.config['trainingsetindex']

        self.project_path = self.config['project_path']
        self.video_path = self.config['video_path']
        self.compressed_cropped_dir_path = os.path.dirname(self.video_path)
        self.clip = vp(fname=self.video_path)

        self._trainFraction = self.config['TrainingFraction'][self.trainingsetindex]
        self._DLCscorer = auxiliaryfunctions.GetScorerName(
            self.config, self.shuffle, self._trainFraction)

        self.label_path = self.video_path.split(
            '.')[0] + self._DLCscorer + '.h5'

        self.df_label = pd.read_hdf(self.label_path)

        self.df_bodyparts = self.df_label[self._DLCscorer][self.bodyparts]
        self.df_bodyparts_likelihood = self.df_bodyparts.iloc[:, self.df_bodyparts.columns.get_level_values(
            1) == 'likelihood']
        self.df_bodyparts_x = self.df_bodyparts.iloc[:,
                                                     self.df_bodyparts.columns.get_level_values(1) == 'x']
        self.df_bodyparts_y = self.df_bodyparts.iloc[:,
                                                     self.df_bodyparts.columns.get_level_values(1) == 'y']

        self.nx = self.clip.width()
        self.ny = self.clip.height()

        # plotting properties
        self._dotsize = 7
        self._line_thickness = 1
        self._pcutoff = self.config['pcutoff']
        self._colormap = self.config['colormap']
        self._label_colors = get_cmap(len(self.bodyparts), name=self._colormap)
        self._alphavalue = self.config['alphavalue']
        self._fig_size = [12, 8]
        self._dpi = 100

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

    @property
    def colormap(self):
        return self._colormap

    @colormap.setter
    def colormap(self, value):
        if isinstance(value, str):
            self._colormap = value
            self._label_colors = get_cmap(
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

    def plot_core(self, fig, ax, frame_num):
        # it's given in 3 channels but every channel is the same i.e. grayscale
        image = self.clip._read_specific_frame(frame_num)

        ax_frame = ax.imshow(image, cmap='gray')

        # plot bodyparts above the pcutoff
        bpindex, df_x_coords, df_y_coords = self.coords_pcutoff(frame_num)
        ax_scatter = ax.scatter(df_x_coords.values, df_y_coords.values, s=self.dotsize**2,
                                color=self._label_colors(bpindex), alpha=self.alphavalue)

        return {'ax_frame': ax_frame, 'ax_scatter': ax_scatter}

    def plot_one_frame(self, frame_num, save_fig=False):

        fig, ax = self.configure_plot()

        ax_dict = self.plot_core(fig, ax, frame_num)

        plt.axis('off')
        plt.title('frame num: ' + str(frame_num), fontsize=30)
        plt.tight_layout()

        fig.canvas.draw()

        if save_fig:
            plt.savefig(os.path.join(
                self.video_path.split('.')[0] + '_frame_' + str(frame_num) + '.png'))

        # return ax_dict

    def plot_multi_frames(self, start, end, save_gif=False):

        plt_list = []

        fig, ax = self.configure_plot()

        for frame_num in range(start, end):
            ax_dict = self.plot_core(fig, ax, frame_num)

            plt.axis('off')
            plt.title('frame num: ' + str(frame_num), fontsize=30)
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


class PupilFitting(PlotBodyparts):
    # for this class, all bodyparts must be provided... so why bother providing bodyparts as input?
    def __init__(self, config, bodyparts='all'):
        """
        Input:
            config: dictionary
                A dictionary that contains animal id, session, scan idx, and a path to config
            bodyparts: list
                A list that contains bodyparts to plot. Each bodypart is in a string format. If none provided,
                then by default it plots ALL existing bodyplots in config.yaml file.

        """
        super().__init__(config, bodyparts=bodyparts)

        self.complete_eyelid_graph = {'eyelid_top': 'eyelid_top_right',
                                      'eyelid_top_right': 'eyelid_right',
                                      'eyelid_right': 'eyelid_right_bottom',
                                      'eyelid_right_bottom': 'eyelid_bottom',
                                      'eyelid_bottom': 'eyelid_bottom_left',
                                      'eyelid_bottom_left': 'eyelid_left',
                                      'eyelid_left': 'eyelid_left_top',
                                      'eyelid_left_top': 'eyelid_top'}

        self._circle_threshold = 3
        self._ellipse_threshold = 6

    @property
    def circle_threshold(self):
        return self._ellipse_threshold

    @circle_threshold.setter
    def circle_threshold(self, value):
        if value > 8:
            raise ValueError("value must be equal to or less than 8!")
        else:
            self._ellipse_threshold = value

    @property
    def ellipse_threshold(self):
        return self._ellipse_threshold

    @ellipse_threshold.setter
    def ellipse_threshold(self, value):
        if value > 8:
            raise ValueError("value must be equal to or less than 8!")
        else:
            self._ellipse_threshold = value

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
                frame: a numpy array of the frame with pupil circle
                center: coordinates of the center of the fitted circle. In tuple format
                radius: radius of the fitted circle in int format
                pupil_labels_num: number of pupil labels used for fitting
                mask: a binary mask for the fitted circle area
        """

        mask = np.zeros(frame.shape, dtype=np.uint8)

        _, df_x_coords, df_y_coords = self.coords_pcutoff(frame_num)

        pupil_labels = [label for label in list(
            df_x_coords.index.get_level_values(0)) if 'pupil' in label]

        if len(pupil_labels) < self.circle_threshold:
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
                               radius=int(round(radius)), color=(0, 255, 0), thickness=self.line_thickness)

            mask = cv2.circle(img=mask, center=(int(round(x)), int(round(y))),
                              radius=int(round(radius)), color=(0, 255, 0), thickness=self.line_thickness)

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
            A dictionary with the fitted frame, center and radius of the fitted circle. If fitting did
            not occur, return the original frame with center and raidus as None.
            For each key in dictionary:
                frame: a numpy array of the frame with pupil circle
                center: coordinates of the center of the fitted circle. In tuple format
                radius: radius of the fitted circle in int format
                pupil_labels_num: number of pupil labels used for fitting
                mask: a binary mask for the fitted circle area
        """

        mask = np.zeros(frame.shape, dtype=np.uint8)

        _, df_x_coords, df_y_coords = self.coords_pcutoff(frame_num)

        pupil_labels = [label for label in list(
            df_x_coords.index.get_level_values(0)) if 'pupil' in label]

        if len(pupil_labels) < self.ellipse_threshold:
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
            frame = cv2.ellipse(np.array(frame), rotated_rect, color=(
                0, 0, 255), thickness=self.line_thickness)
            mask = cv2.ellipse(np.array(frame), rotated_rect, color=(
                0, 0, 255), thickness=self.line_thickness)

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
                color_mask: numpy array
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
            threshold = self.circle_threshold

        elif fitting_method == 'ellipse':
            threshold = self.ellipse_threshold

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

        return dict(color_mask=color_mask, visible_portion=visible_portion)

    def fitted_core(self, fig, ax, frame_num, fitting_method=None):
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

        # plot bodyparts above the pcutoff
        bpindex, x_coords, y_coords = self.coords_pcutoff(frame_num)
        ax_scatter = ax.scatter(x_coords.values, y_coords.values, s=self.dotsize**2,
                                color=self._label_colors(bpindex), alpha=self.alphavalue)

        eyelid_connected = self.connect_eyelids(frame_num, frame=image)

        circle_fit = self.fit_circle_to_pupil(
            frame_num, frame=eyelid_connected['frame'])

        ellipse_fit = self.fit_ellipse_to_pupil(
            frame_num, eyelid_connected['frame'], threshold=self.threshold)

        circle_visible = self.detect_visible_pupil_area(eyelid_connect_dict=eyelid_connected,
                                                        fit_dict=circle_fit,
                                                        fitting_method='circle')
        
        ellipse_visible = self.detect_visible_pupil_area(eyelid_connect_dict=eyelid_connected,
                                                        fit_dict=ellipse_fit,
                                                        fitting_method='ellipse')

        circle_frame = ax.imshow(circle_fit['frame'])
        ellipse_frame = ax.imshow(ellipse_fit['frame'])
        

        return {'ax_frame': ax_frame, 'ax_scatter': ax_scatter, 'ax_mask': ax_mask}

    def plot_fitted_frame(self, frame_num, save_fig=False):

        fig, ax = self.configure_plot()
        ax_dict = self.fitted_plot_core(fig, ax, frame_num)

        plt.title('frame num: ' + str(frame_num), fontsize=30)

        plt.axis('off')
        plt.tight_layout()

        fig.canvas.draw()

        if save_fig:
            plt.savefig(os.path.join(
                self.compressed_cropped_dir_path, 'fitted_frame_' + str(frame_num) + '.png'))

    def plot_fitted_multi_frames(self, start, end, save_gif=False):

        fig, ax = self.configure_plot()

        plt_list = []

        for frame_num in range(start, end):

            _ = self.fitted_plot_core(fig, ax, frame_num)

            plt.axis('off')
            plt.title('frame num: ' + str(frame_num), fontsize=30)
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
            gif_path = self.video_path.split('.')[0] + '_fitted_' + \
                str(start) + '_' + str(end) + '.gif'
            imageio.mimsave(gif_path, plt_list, fps=1)

        plt.close('all')

    def make_movie(self, start, end):

        import matplotlib.animation as animation

        # initlize with start frame
        fig, ax = self.configure_plot()
        # ax_dict = self.fitted_plot_core(fig, ax, frame_num=start)
        _ = self.fitted_plot_core(fig, ax, frame_num=start)

        plt.axis('off')
        plt.title('frame num: ' + str(start), fontsize=30)
        plt.tight_layout()

        def update_frame(frame_num):

            # clear out the axis
            plt.cla()
            # new_ax_dict = self.fitted_plot_core(fig, ax, frame_num=frame_num)
            _ = self.fitted_plot_core(fig, ax, frame_num=frame_num)

            plt.axis('off')
            plt.tight_layout()
            plt.title('frame num: ' + str(frame_num), fontsize=30)

        ani = animation.FuncAnimation(fig, update_frame, range(
            start+1, end))  # , interval=int(1/self.clip.FPS)
        # ani = animation.FuncAnimation(fig, self.plot_fitted_frame, 10)
        writer = animation.writers['ffmpeg'](fps=self.clip.FPS)

        # dpi=self.dpi, fps=self.clip.FPS
        video_name = os.path.join(
            self.path_to_analysis, self._case_full_name + '_labeled.avi')
        ani.save(video_name, writer=writer, dpi=self.dpi)

        return ani
