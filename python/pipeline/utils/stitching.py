import numpy as np
from scipy.ndimage import interpolation

class Position:
    NONCONTIGUOUS = 0
    LEFT = 1
    RIGHT = 2


class _ROI():
    """ Simple class to hold ROI coordinates. """
    def __init__(self, id_, x, y, z):
        self.id = id_
        self.x = x
        self.y = y
        self.z = z

    def rot90(self):
        self.x, self.y = -self.y, self.x # parallel assignment

    def rot270(self):
        self.x, self.y = self.y, -self.x # parallel assignment


class StitchedROI():
    """ Receives ROIs and its coordinates and blends them into a single volume. """
    def __init__(self, roi, x, y, z, id_, dtype=np.float32):
        self.volume = roi.astype(dtype, copy=False)
        self.dtype = dtype
        self.x = x
        self.y = y
        self.z = z
        self.mask = np.ones([self.height, self.width], dtype=np.float32)
        self.roi_coordinates = [_ROI(id_, x, y, z)] # bookkeeping

    @property
    def height(self):
        return self.volume.shape[1]

    @property
    def width(self):
        return self.volume.shape[2]

    @property
    def depth(self):
        return self.volume.shape[0]

    def left_or_right(self, roi, rel_tol=0.15):
        """ Whether ROI is to the right, left or noncontiguous to the current volume.

        To be contiguous, ROIs have to start at the same z (up to some tolerance), have
        have equal depth, equal height (up to some tolerance) and overlap in one of the
        two sides.

        :returns: Position.RIGHT/Position.LEFT if roi is to the right/left of the volume.
            Position.NONCONTIGUOUS, otherwise
        """
        position = Position.NONCONTIGUOUS
        if (self.depth == roi.depth and abs(self.z - roi.z) < rel_tol * self.depth and
            abs(self.y - roi.y) < rel_tol * self.height and abs(self.height - roi.height)
            < rel_tol * self.height):
            if (roi.x - roi.width / 2) < (self.x + self.width / 2):
                position = Position.RIGHT
            elif (roi.x + roi.width / 2) > (self.x - self.width / 2):
                position = Position.LEFT

        return position

    def is_aside_to(self, roi):
        side = self.left_or_right(roi)
        return side == Position.LEFT or side == Position.RIGHT

    def join_with(self, roi, x, y, smooth_blend=True):
        """ Appends a new ROI to this volume at the given coordinates.

        :param StitchedROI roi: The other roi
        :param float x, y: Coordinates of the other ROI
        :param bool smooth_blend: Whether to taper edges for a smoother blending. It
            assumes roi is to the right of self.
        """
        # Compute size of the joint ROI
        x_min = min(self.x - self.width / 2, x - roi.width / 2)
        x_max = max(self.x + self.width / 2, x + roi.width / 2)
        y_min = min(self.y - self.height / 2, y - roi.height / 2)
        y_max = max(self.y + self.height / 2, y + roi.height / 2)
        x_max -= (x_max - x_min) % 1 # Make sure they add up to an integer value
        y_max -= (y_max - y_min) % 1 # Make sure they add up to an integer value
        output_height, output_width = int(round(y_max - y_min)), int(round(x_max - x_min))

        # Taper sides for smoother blending
        if smooth_blend:
            from scipy.signal import hann

            taper_size = int(round((self.x + self.width / 2) - (x - roi.width / 2))) # overlap
            if taper_size < 1:
                raise ValueError('Smooth blending expects input ROI to be to the right '
                                 'of self: left.join_with(right)')
            taper = hann(2 * taper_size)[:taper_size]

            self.mask[..., -taper_size:] *= (1 - taper)
            roi.mask[..., :taper_size] *= taper

        # Initialize empty (big) rois
        mask1 = np.zeros([output_height, output_width], dtype=np.float32)
        roi1 = np.zeros([self.depth, output_height, output_width], dtype=self.dtype)
        mask1[:self.height, :self.width] = self.mask
        roi1[:, :self.height, :self.width] = self.volume

        mask2 = np.zeros([output_height, output_width], dtype=np.float32)
        roi2 = np.zeros([roi.depth, output_height, output_width], dtype=roi.dtype)
        mask2[:roi.height, :roi.width] = roi.mask
        roi2[:, :roi.height, :roi.width] = roi.volume

        # Move rois to their final position
        delta_x1, delta_y1 = (self.x - self.width / 2) - x_min, (self.y + self.height / 2) - y_max
        interpolation.shift(mask1, (-delta_y1, delta_x1), output=mask1)
        interpolation.shift(roi1, (0, -delta_y1, delta_x1), output=roi1)

        delta_x2, delta_y2 = (x - roi.width / 2) - x_min, (y + roi.height / 2) - y_max
        interpolation.shift(mask2, (-delta_y2, delta_x2), output=mask2)
        interpolation.shift(roi2, (0, -delta_y2, delta_x2), output=roi2)

        # Blend (mask act as weights and normalization needed for them to sum to 1)
        self.mask = mask1 + mask2
        self.volume = roi1 * mask1 + roi2 * mask2
        self.volume[:, self.mask > 1e-7] /= self.mask[self.mask > 1e-7]

        # Update coordinates (bookkeeping)
        self.x = x_min + output_width / 2
        self.y = y_min + output_height / 2
        for roi_coord in roi.roi_coordinates:
            roi_coord.x += (x - roi_coord.x)
            roi_coord.y += (y - roi_coord.y)
            self.roi_coordinates.append(roi_coord)

    def rot90(self):
        """ Rotates volume 90 degrees over the z-axis counterclockwise. """
        self.x, self.y = -self.y, self.x
        self.mask = np.rot90(self.mask)
        self.volume = np.rot90(self.volume, axes=(1, 2))
        for roi_coord in self.roi_coordinates:
            roi_coord.rot90()

    def rot270(self):
        """ Inverse of rot90. """
        self.x, self.y = self.y, -self.x
        self.mask = np.rot90(self.mask, k=3)
        self.volume = np.rot90(self.volume, k=3, axes=(1, 2))
        for roi_coord in self.roi_coordinates:
            roi_coord.rot270()

    def trim(self):
        """ Trims black edges of the volume. """
        # Trim lines (rows or columns) that have black pixels in the corners.
        self._trim_corners()
        self.rot90()
        self._trim_corners()
        self.rot270()

        # Trim remaining lines that have black pixels in the middle
        self._trim_centers()

    def _trim_corners(self):
        """ Trim lines with black spaces in upper right and lower left corner. """
        binary_mask = self.mask > 1e-7
        if not binary_mask[0, 0]: # either delete first row or first column
            row_blacks = np.logical_not(binary_mask[0]).sum() - binary_mask[0].sum()
            col_blacks = np.logical_not(binary_mask[:, 0]).sum() - binary_mask[:, 0].sum()
            if row_blacks >= col_blacks:
                self.mask = self.mask[1:]
                self.volume = self.volume[:, 1:, :]
                self.y -= 0.5
            else:
                self.mask = self.mask[:, 1:]
                self.volume = self.volume[:, :, 1:]
                self.x += 0.5
            self._trim_corners() # keep trimming recursively
        elif not binary_mask[-1, -1]: # either delete last row or last column
            row_blacks = np.logical_not(binary_mask[-1]).sum() - binary_mask[-1].sum()
            col_blacks = np.logical_not(binary_mask[:, -1]).sum() - binary_mask[:, -1].sum()
            if row_blacks >= col_blacks:
                self.mask = self.mask[:-1]
                self.volume = self.volume[:, :-1, :]
                self.y += 0.5
            else:
                self.mask = self.mask[:, :-1]
                self.volume = self.volume[:, :, :-1]
                self.x -= 0.5
            self._trim_corners() # keep trimming recursively

    def _trim_centers(self):
        """ Trim lines with black spaces in the middle. Call after _trim_corners."""
        binary_mask = self.mask < 1e-7 # True in black spaces
        if binary_mask[0].sum() > 0: # black space in first row
            self.mask = self.mask[1:]
            self.volume = self.volume[:, 1:, :]
            self.y -= 0.5
            self._trim_centers()
        if binary_mask[-1].sum() > 0: # black space in last row
            self.mask = self.mask[:-1]
            self.volume = self.volume[:, :-1, :]
            self.y += 0.5
            self._trim_centers()
        if binary_mask[:, 0].sum() > 0: # black space in first column
            self.mask = self.mask[:, 1:]
            self.volume = self.volume[:, :, 1:]
            self.x += 0.5
            self._trim_centers()
        if binary_mask[:, -1].sum() > 0: # black space in last row
            self.mask = self.mask[:, :-1]
            self.volume = self.volume[:, :, :-1]
            self.x -= 0.5
            self._trim_centers()


def linear_stitch(left, right, max_overlap=0.5):
    """ Compute offsets to stitch two side-by-side volumes via cross-correlation.

    Subpixel shifts are calculated per depth and the median is returned.

    Arguments:
    :param left: Volume (depth x height x width) to be stitched in the left.
    :param right: Volume (depth x height x width) to be stitched in the right.
    :param max_overlap: Maximum horizontal overlap allowed after stitching. Expressed as
        a proportion of the smaller roi width.

    :returns: (delta_x, delta_y). Distance between center of right ROI and left ROI after
        stitching (right_center - left_center)
    """
    from skimage.feature import match_template

    # Get some params
    num_slices, right_height, right_width = right.shape
    num_slices, left_height, left_width = left.shape

    # Get template (leftmost 10% of the right volume)
    template_width = int(round(0.1 * min(right_width, left_width)))
    template_height = min(right_height, left_height)
    template_width -= 1 if template_width % 2 == 0 else 0 # make odd
    template_height -= 1 if template_height % 2 == 0 else 0 # make odd
    template = right[:, :template_height, :template_width]

    # Restrict left strip to the desired overlap (computationally stable and efficient)
    max_overlap_in_pixels = int(round(max_overlap * min(right_width, left_width)))
    max_overlap_in_pixels -= template_width // 2
    left_strip = left[..., -max_overlap_in_pixels:]

    # Compute cross-correlation
    corr = np.zeros(left_strip.shape[-2:])
    for l, r in zip(left_strip, template):
        corr += match_template(l, r, pad_input=True, constant_values=np.mean(l))
    corr /= num_slices

    # Get best match
    right_y, right_x = np.unravel_index(np.argmax(corr), corr.shape)

    # Compute right_center - left_center (a bit confusing but they are right)
    right_y += 1 if min(right_height, left_height) % 2 == 0 else 0.5 # 0.5 to move from index to middle of pixel (0->0.5) and 0.5 if height was even
    delta_y = -(right_y - left_height / 2) # negative to change direction of y axis

    x_shift = (left_strip.shape[-1] - right_x)  + template_width // 2 # how much the edge of template moved inside left
    delta_x = right_width / 2 + left_width / 2 - x_shift

    return delta_x, delta_y