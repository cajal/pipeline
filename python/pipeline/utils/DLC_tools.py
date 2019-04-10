
import math
import os
import yaml
import cv2
from pathlib import Path
import ruamel.yaml

from deeplabcut.utils import auxiliaryfunctions

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


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
