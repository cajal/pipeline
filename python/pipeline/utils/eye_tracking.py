from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from ..exceptions import PipelineException

try:
    import cv2
except ImportError:
    print("Could not find cv2. You won't be able to use the pupil tracker.")

ANALOG_PACKET_LEN = 2000


class CVROIGrabber:
    start = None
    end = None
    roi = None

    def __init__(self, img):
        self.img = img
        self.draw_img = np.asarray(img / img.max(), dtype=float)
        self.mask = 1 + 0 * img
        self.exit = False
        self.r = 40
        self.X, self.Y = np.mgrid[:img.shape[0], :img.shape[1]]

    def grab(self):
        print('Contrast (std)', np.std(self.img))
        img = np.asarray(self.img / self.img.max(), dtype=float)
        cv2.namedWindow('real image')
        cv2.setMouseCallback('real image', self, 0)

        while not self.exit:
            cv2.imshow('real image', img)
            if (cv2.waitKey(0) & 0xFF) == ord('q'):
                cv2.waitKey(1)
                cv2.destroyAllWindows()
                break
        cv2.waitKey(2)

    def __call__(self, event, x, y, flags, params):
        # img = np.asarray(self.img , dtype=np.uint8)[...,None] * np.ones((1,1,3), dtype=np.uint8)
        img = np.asarray(self.img / self.img.max(), dtype=float)
        cv2.imshow('real image', self.draw_img)

        if event == cv2.EVENT_LBUTTONDOWN:
            print('Start Mouse Position: ' + str(x) + ', ' + str(y))
            self.start = np.asarray([x, y])

        elif event == cv2.EVENT_LBUTTONUP:
            self.end = np.asarray([x, y])
            x = np.vstack((self.start, self.end))
            tmp = np.hstack((x.min(axis=0), x.max(axis=0)))
            roi = np.asarray([[tmp[1], tmp[3]], [tmp[0], tmp[2]]], dtype=int) + 1
            crop = img[roi[0, 0]:roi[0, 1], roi[1, 0]:roi[1, 1]]
            crop = np.asarray(crop / crop.max(), dtype=float)
            self.roi = roi
            cv2.imshow('crop', crop)

            # m = (img * self.mask).copy() # needed for a weird reason
            self.draw_img = (img * self.mask).copy()
            cv2.rectangle(self.draw_img, tuple(self.start), tuple(self.end), (0, 255, 0), 2)

            cv2.imshow('real image', self.draw_img)
            key = (cv2.waitKey(0) & 0xFF)
            if key == ord('q'):
                cv2.destroyAllWindows()
                self.exit = True
            elif key == ord('c'):
                self.mask = 0 * self.mask + 1

        elif event == cv2.EVENT_MBUTTONDOWN:
            img = np.asarray(self.img / self.img.max(), dtype=float)

            self.mask[(self.X - y) ** 2 + (self.Y - x) ** 2 < self.r ** 2] = 0.
            self.draw_img[(self.X - y) ** 2 + (self.Y - x) ** 2 < self.r ** 2] = 0.
            cv2.imshow('real image', self.draw_img)

            key = (cv2.waitKey(0) & 0xFF)
            if key == ord('q'):
                cv2.destroyAllWindows()
                self.exit = True
            elif key == ord('c'):
                self.mask = 0 * self.mask + 1


class ROIGrabber:
    """
    Interactive matplotlib figure to grab an ROI from an image.

    Usage:

    rg = ROIGrabber(img)
    # select roi
    print(rg.roi) # get ROI
    """

    def __init__(self, img):
        plt.switch_backend('GTK3Agg')
        self.img = img
        self.start = None
        self.current = None
        self.end = None
        self.pressed = False
        self.fig, self.ax = plt.subplots(facecolor='w')

        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.replot()
        plt.show(block=True)

    def draw_rect(self, fr, to, color='dodgerblue'):
        x = np.vstack((fr, to))
        fr = x.min(axis=0)
        to = x.max(axis=0)
        self.ax.plot(fr[0] * np.ones(2), [fr[1], to[1]], color=color, lw=2)
        self.ax.plot(to[0] * np.ones(2), [fr[1], to[1]], color=color, lw=2)
        self.ax.plot([fr[0], to[0]], fr[1] * np.ones(2), color=color, lw=2)
        self.ax.plot([fr[0], to[0]], to[1] * np.ones(2), color=color, lw=2)
        self.ax.plot(fr[0], fr[1], 'ok', mfc='gold')
        self.ax.plot(to[0], to[1], 'ok', mfc='deeppink')

    def replot(self):
        self.ax.clear()
        self.ax.imshow(self.img, cmap=plt.cm.gray)

        if self.pressed:
            self.draw_rect(self.start, self.current, color='lime')
        elif self.start is not None and self.end is not None:
            self.draw_rect(self.start, self.current)
        self.ax.axis('tight')
        self.ax.set_aspect(1)
        self.ax.set_title('Close window when done', fontsize=16, fontweight='bold')
        plt.draw()

    @property
    def roi(self):
        x = np.vstack((self.start, self.end))
        tmp = np.hstack((x.min(axis=0), x.max(axis=0)))
        return np.asarray([[tmp[1], tmp[3]], [tmp[0], tmp[2]]], dtype=int) + 1

    def on_press(self, event):
        if event.xdata is not None and event.ydata is not None:
            self.pressed = True
            self.start = np.asarray([event.xdata, event.ydata])

    def on_release(self, event):
        if event.xdata is not None and event.ydata is not None:
            self.end = np.asarray([event.xdata, event.ydata])
        else:
            self.end = self.current
        self.pressed = False
        self.replot()

    def on_move(self, event):
        if event.xdata is not None and event.ydata is not None:
            self.current = np.asarray([event.xdata, event.ydata])
            if self.pressed:
                self.replot()


class PupilTracker:
    """
    Parameters:

    perc_high                    : float        # upper percentile for bright pixels
    perc_low                     : float        # lower percentile for dark pixels
    perc_weight                  : float        # threshold will be perc_weight*perc_low + (1- perc_weight)*perc_high
    relative_area_threshold      : float        # enclosing rotating rectangle has to have at least that amount of area
    ratio_threshold              : float        # ratio of major and minor radius cannot be larger than this
    error_threshold              : float        # threshold on the RMSE of the ellipse fit
    min_contour_len              : int          # minimal required contour length (must be at least 5)
    margin                       : float        # relative margin the pupil center should not be in
    contrast_threshold           : float        # contrast below that threshold are considered dark
    speed_threshold              : float        # eye center can at most move that fraction of the roi between frames
    dr_threshold                 : float        # maximally allow relative change in radius

    """

    def __init__(self, param, mask=None):
        self._params = param
        self._center = None
        self._radius = None
        self._mask = mask
        self._last_detection = 1

    @staticmethod
    def goodness_of_fit(contour, ellipse):
        center, size, angle = ellipse
        angle *= np.pi / 180
        err = 0
        for coord in contour.squeeze().astype(np.float):
            posx = (coord[0] - center[0]) * np.cos(-angle) - (coord[1] - center[1]) * np.sin(-angle)
            posy = (coord[0] - center[0]) * np.sin(-angle) + (coord[1] - center[1]) * np.cos(-angle)
            err += ((posx / size[0]) ** 2 + (posy / size[1]) ** 2 - 0.25) ** 2

        return np.sqrt(err / len(contour))

    @staticmethod
    def restrict_to_long_axis(contour, ellipse, corridor):
        center, size, angle = ellipse
        angle *= np.pi / 180
        R = np.asarray([[np.cos(-angle), - np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])
        contour = np.dot(contour.squeeze() - center, R.T)
        contour = contour[np.abs(contour[:, 0]) < corridor * ellipse[1][1] / 2]
        return (np.dot(contour, R) + center).astype(np.int32)

    def get_pupil_from_contours(self, contours, small_gray, mask, show_matching=5):
        ratio_thres = self._params['ratio_threshold']
        area_threshold = self._params['relative_area_threshold']
        error_threshold = self._params['error_threshold']
        min_contour = self._params['min_contour_len']
        margin = self._params['margin']
        speed_thres = self._params['speed_threshold']
        dr_thres = self._params['dr_threshold']
        err = np.inf
        best_ellipse = None
        best_contour = None
        results, cond = defaultdict(list), defaultdict(list)
        kernel = np.ones((3, 3))
        for j, cnt in enumerate(contours):

            mask2 = cv2.erode(mask, kernel, iterations=1)
            idx = mask2[cnt[..., 1], cnt[..., 0]] > 0
            cnt = cnt[idx]

            if len(cnt) < min_contour:  # otherwise fitEllipse won't work
                continue

            ellipse = cv2.fitEllipse(cnt)
            ((x, y), axes, angle) = ellipse
            if min(axes) == 0:  # otherwise ratio won't work
                continue
            ratio = max(axes) / min(axes)
            area = np.prod(ellipse[1]) / np.prod(small_gray.shape)
            curr_err = self.goodness_of_fit(cnt, ellipse)

            results['ratio'].append(ratio)
            results['area'].append(area)
            results['rmse'].append(curr_err)
            results['x coord'].append(x / small_gray.shape[1])
            results['y coord'].append(y / small_gray.shape[0])

            center = np.array([x / small_gray.shape[1], y / small_gray.shape[0]])
            r = max(axes)

            dr = 0 if self._radius is None else np.abs(r - self._radius) / self._radius
            dx = 0 if self._center is None else np.sqrt(np.sum((center - self._center) ** 2))

            results['dx'].append(dx)
            results['dr/r'].append(dr)
            matching_conditions = 1 * (ratio <= ratio_thres) + 1 * (area >= area_threshold) \
                                  + 1 * (curr_err < error_threshold) \
                                  + 1 * (margin < center[0] < 1 - margin) \
                                  + 1 * (margin < center[1] < 1 - margin) \
                                  + 1 * (dx < speed_thres * self._last_detection) \
                                  + 1 * (dr < dr_thres * self._last_detection)
            cond['ratio'].append(ratio <= ratio_thres)
            cond['area'].append(area >= area_threshold)
            cond['rmse'].append(curr_err < error_threshold)
            cond['x coord'].append(margin < center[0] < 1 - margin)
            cond['y coord'].append(margin < center[1] < 1 - margin)
            cond['dx'].append(dx < speed_thres * self._last_detection)
            cond['dr/r'].append(dr < dr_thres * self._last_detection)

            results['conditions'] = matching_conditions
            cond['conditions'].append(True)

            if curr_err < err and matching_conditions == 7:
                best_ellipse = ellipse
                best_contour = cnt
                err = curr_err
                cv2.ellipse(small_gray, ellipse, (0, 0, 255), 2)
            elif matching_conditions >= show_matching:
                cv2.ellipse(small_gray, ellipse, (255, 0, 0), 2)

        if best_ellipse is None:
            df = pd.DataFrame(results)
            df2 = pd.DataFrame(cond)

            print('-', end="", flush=True)
            if 'conditions' in df.columns and np.any(df['conditions'] >= show_matching):
                idx = df['conditions'] >= show_matching
                df = df[idx]
                df2 = df2[idx]
                df[df2] = np.nan
                print("\n", df, flush=True)
            self._last_detection += 1
        else:
            self._last_detection = 1

        return best_contour, best_ellipse

    _running_avg = None

    def preprocess_image(self, frame, eye_roi):
        h = int(self._params['gaussian_blur'])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_std = np.std(gray)

        small_gray = gray[slice(*eye_roi[0]), slice(*eye_roi[1])]

        # Manual meso settins
        if 'extreme_meso' in self._params and self._params['extreme_meso']:
            c = 0.1
            p = 7
            if self._running_avg is None:
                self._running_avg = np.array(small_gray) ** p
            else:
                self._running_avg = c * np.array(small_gray) ** p + (1 - c) * self._running_avg
                small_gray += self._running_avg.astype(np.uint8) - small_gray  # big hack
        # --- mesosetting end

        blur = cv2.GaussianBlur(small_gray, (2 * h + 1, 2 * h + 1), 0)  # play with blur

        _, thres = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return gray, small_gray, img_std, thres, blur

    @staticmethod
    def display(gray, blur, thres, eye_roi, fr_count, n_frames, ncontours=0, contour=None, ellipse=None,
                eye_center=None,
                font=cv2.FONT_HERSHEY_SIMPLEX):
        cv2.imshow('blur', blur)
        cv2.imshow('threshold', thres)
        cv2.putText(gray, "Frames {fr_count}/{frames} | Found contours {ncontours}".format(fr_count=fr_count,
                                                                                           frames=n_frames,
                                                                                           ncontours=ncontours),
                    (10, 30), font, 1, (255, 255, 255), 2)

        if contour is not None and ellipse is not None and eye_center is not None:
            ellipse = list(ellipse)
            ellipse[0] = tuple(eye_center)
            ellipse = tuple(ellipse)
            cv2.drawContours(gray, [contour], 0, (255, 0, 0), 1, offset=tuple(eye_roi[::-1, 0]))
            cv2.ellipse(gray, ellipse, (0, 0, 255), 2)
            epy, epx = np.round(eye_center).astype(int)
            gray[epx - 3:epx + 3, epy - 3:epy + 3] = 0
        cv2.imshow('frame', gray)

    def track(self, videofile, eye_roi, display=False):
        contrast_low = self._params['contrast_threshold']

        print("Tracking videofile", videofile)
        cap = cv2.VideoCapture(videofile)
        traces = []

        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fr_count = 0
        if self._mask is not None:
            small_mask = self._mask[slice(*eye_roi[0]), slice(*eye_roi[1])].squeeze()
        else:
            small_mask = np.ones(np.diff(eye_roi, axis=1).squeeze().astype(int), dtype=np.uint8)

        while cap.isOpened():
            if fr_count >= n_frames:
                print("Reached end of videofile ", videofile)
                break

            # --- read frame
            ret, frame = cap.read()
            fr_count += 1

            # --- if we don't get a frame, don't add any tracking results
            if not ret:
                traces.append(dict(frame_id=fr_count))
                continue

            # --- print out if there's not display
            if fr_count % 500 == 0:
                print("\tframe ({}/{})".format(fr_count, n_frames))

            # --- preprocess and treshold images
            gray, small_gray, img_std, thres, blur = self.preprocess_image(frame, eye_roi)

            # --- if contrast is too low, skip it
            if img_std < contrast_low:
                traces.append(dict(frame_id=fr_count,
                                   frame_intensity=img_std))
                print('_', end="", flush=True)
                if display:
                    self.display(gray, blur, thres, eye_roi, fr_count, n_frames)
                continue

            # --- detect contours
            ellipse, eye_center, contour = None, None, None

            print(thres.shape, small_mask.shape)
            thres *= small_mask

            _, contours, hierarchy1 = cv2.findContours(thres.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # contour, ellipse = self.get_pupil_from_contours(contours, small_gray)
            contour, ellipse = self.get_pupil_from_contours(contours, blur, small_mask)
            # if display:
            #     self.display(self._mask.copy() if self._mask is not None else gray,
            #                  blur, thres, eye_roi, fr_count, n_frames, ncontours=len(contours))


            if contour is None:
                traces.append(dict(frame_id=fr_count, frame_intensity=img_std))
            else:
                eye_center = eye_roi[::-1, 0] + np.asarray(ellipse[0])
                self._center = np.asarray(ellipse[0]) / np.asarray(small_gray.shape[::-1])
                self._radius = max(ellipse[1])

                traces.append(dict(center=eye_center,
                                   major_r=np.max(ellipse[1]),
                                   rotated_rect=np.hstack(ellipse),
                                   contour=contour.astype(np.int16),
                                   frame_id=fr_count,
                                   frame_intensity=img_std
                                   ))
            if display:
                self.display(self._mask * gray if self._mask is not None else gray, blur, thres, eye_roi,
                             fr_count, n_frames, ellipse=ellipse,
                             eye_center=eye_center, contour=contour, ncontours=len(contours))
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                raise PipelineException('Tracking aborted')

        cap.release()
        cv2.destroyAllWindows()

        return traces
