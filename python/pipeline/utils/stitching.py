import numpy as np
from scipy import signal
from scipy.ndimage import interpolation
from pipeline.utils import galvo_corrections


class Position:
    NONCONTIGUOUS = 0
    LEFT = 1
    RIGHT = 2


class StitchedSlice():
    """ A single slice from a stitched ROI. """
    def __init__(self, slice_, x, y, dtype):
        self.slice = slice_.astype(dtype, copy=False)
        self.x = x
        self.y = y
        self.dtype=dtype
        self.mask = np.ones_like(self.slice)

    @property
    def height(self):
        return self.slice.shape[0]

    @property
    def width(self):
        return self.slice.shape[1]

    def rot90(self):
        """ Rotates slice 90 degrees counterclockwise. """
        self.x, self.y = -self.y, self.x # parallel assignment
        self.mask = np.rot90(self.mask)
        self.slice = np.rot90(self.slice)

    def join_with(self, other, x, y, smooth_blend=True):
        """ Stitches a new slice to the current slice at the given coordinates.

        :param StitchedSlice other: The other slice.
        :param float xs, ys: Coordinates of the other slices.
        :param bool smooth_blend: Whether to taper edges for a smoother blending. It
            assumes other is aside self (not above or below).
        """
        # Compute size of the joint ROI
        x_min = min(self.x - self.width / 2, x - other.width / 2)
        x_max = max(self.x + self.width / 2, x + other.width / 2)
        y_min = min(self.y - self.height / 2, y - other.height / 2)
        y_max = max(self.y + self.height / 2, y + other.height / 2)
        x_max -= (x_max - x_min) % 1 # Make sure they add up to an integer value
        y_max -= (y_max - y_min) % 1 # Make sure they add up to an integer value
        output_height, output_width = int(round(y_max - y_min)), int(round(x_max - x_min))

        # Taper sides for smoother blending
        if smooth_blend:
            overlap = int(round((self.width + other.width) -
                                (max(self.x + self.width / 2, x + other.width / 2) -
                                 min(self.x - self.width / 2, x - other.width / 2))))
            taper = signal.hann(2 * overlap)[:overlap]

            if self.x + self.width / 2 > x + other.width / 2:  # other | self
                self.mask[..., :overlap] *= taper
                other.mask[..., -overlap:] *= (1 - taper)
            else:
                other.mask[..., :overlap] *= taper
                self.mask[..., -overlap:] *= (1 - taper)

        # Initialize empty (big) slices
        mask1 = np.zeros([output_height, output_width], dtype=np.float32)
        slice1 = np.zeros([output_height, output_width], dtype=self.dtype)
        mask1[:self.height, :self.width] = self.mask
        slice1[:self.height, :self.width] = self.slice

        mask2 = np.zeros([output_height, output_width], dtype=np.float32)
        slice2 = np.zeros([output_height, output_width], dtype=other.dtype)
        mask2[:other.height, :other.width] = other.mask
        slice2[:other.height, :other.width] = other.slice

        # Move rois to their final position
        delta_x1, delta_y1 = (self.x - self.width / 2) - x_min, (self.y + self.height / 2) - y_max
        mask1 = interpolation.shift(mask1, (-delta_y1, delta_x1), order=1)
        slice1 = interpolation.shift(slice1, (-delta_y1, delta_x1), order=1)

        delta_x2, delta_y2 = (x - other.width / 2) - x_min, (y + other.height / 2) - y_max
        mask2 = interpolation.shift(mask2, (-delta_y2, delta_x2), order=1)
        slice2 = interpolation.shift(slice2, (-delta_y2, delta_x2), order=1)

        # Blend (mask act as weights and normalization needed for them to sum to 1)
        self.mask = mask1 + mask2
        self.slice = slice1 * mask1 + slice2 * mask2
        self.slice[self.mask > 1e-7] /= self.mask[self.mask > 1e-7]

        # Bookkeeping: Update coordinates
        self.x = x_min + output_width / 2
        self.y = y_min + output_height / 2


class ROICoordinates():
    """ Simple class to hold ROI id and coordinates. """
    def __init__(self, id_, xs, ys):
        self.id = id_
        self.xs = xs
        self.ys = ys

    def rot90(self):
        self.xs, self.ys = [-y for y in self.ys], self.xs # parallel assignment


class StitchedROI():
    """ Set of stitched slices that form a volume."""
    def __init__(self, roi, x, y, z, id_, dtype=np.float32):
        self.slices = []
        xs = x if isinstance(x, list) else [x] * roi.shape[0]
        ys = y if isinstance(x, list) else [y] * roi.shape[0]
        for slice_, x, y in zip(roi, xs, ys):
            self.slices.append(StitchedSlice(slice_, x, y, dtype))
        self.z = z
        self.dtype = dtype
        self.roi_coordinates = [ROICoordinates(id_, xs, ys)]  # bookkeeping

    @property
    def height(self):
        y_min = min([slice_.y - slice_.height / 2 for slice_ in self.slices])
        y_max = max([slice_.y + slice_.height / 2 for slice_ in self.slices])
        y_max -= (y_max - y_min) % 1 # Make sure they add up to an integer value
        return int(round(y_max - y_min))

    @property
    def width(self):
        x_min = min([slice_.x - slice_.width / 2 for slice_ in self.slices])
        x_max = max([slice_.x + slice_.width / 2 for slice_ in self.slices])
        x_max -= (x_max - x_min) % 1 # Make sure they add up to an integer value
        return int(round(x_max - x_min))

    @property
    def depth(self):
        return len(self.slices)

    @property
    def x(self):
        x_min = min([slice_.x - slice_.width / 2 for slice_ in self.slices])
        return x_min + self.width / 2

    @property
    def y(self):
        y_min = min([slice_.y - slice_.height / 2 for slice_ in self.slices])
        return y_min + self.height / 2

    @property
    def volume(self):
        """ Collects all slices into a single 3-d volume."""
        x_min = min([slice_.x - slice_.width / 2 for slice_ in self.slices])
        y_min = min([slice_.y - slice_.height / 2 for slice_ in self.slices])
        y_max = max([slice_.y + slice_.height / 2 for slice_ in self.slices])
        y_max -= (y_max - y_min) % 1 # Make sure they add up to an integer value

        # Move each slice to the right position
        volume = np.zeros([self.depth, self.height, self.width], dtype=self.dtype)
        for i, slice_ in enumerate(self.slices):
            volume[i, :slice_.height, :slice_.width] = slice_.slice
            delta_y = (slice_.y + slice_.height / 2) - y_max
            delta_x = (slice_.x - slice_.width / 2) - x_min
            volume[i] = interpolation.shift(volume[i], (-delta_y, delta_x), order=1)

        return volume

    def left_or_right(self, other, rel_tol=0.1, minimum_overlap=25):
        """ Whether the other ROI is to the right, left or noncontiguous to self.

        To be contiguous, ROIs need to start at the same z (up to some tolerance), overlap
        in one of the two sides and have equal depth and height (up to some tolerance).

        :param rel_tol: Percentage of depth/height that z/y could be off.
        :param minimum_overlap: Minimum required amount of overlap in pixels.

        :returns: Position.RIGHT/Position.LEFT if roi is to the right/left of the volume.
            Position.NONCONTIGUOUS, otherwise
        """
        position = Position.NONCONTIGUOUS
        same_depth = (self.depth == other.depth and
                      abs(self.z - other.z) < rel_tol * self.depth)
        same_height = (abs(self.height - other.height) < rel_tol * self.height and
                      abs(self.y - other.y) < rel_tol * self.height)
        overlap = (max(self.x + self.width / 2, other.x + other.width / 2) -
                   min(self.x - self.width / 2, other.x - other.width / 2) +
                   minimum_overlap < self.width + other.width)
        if same_depth and same_height and overlap:
            if self.x + self.width / 2 > other.x + other.width / 2:
                position = Position.LEFT
            else:
                position = Position.RIGHT

        return position

    def is_aside_to(self, other):
        side = self.left_or_right(other)
        return side == Position.LEFT or side == Position.RIGHT

    def rot90(self):
        """ Rotates volume 90 degrees over the z-axis counterclockwise. """
        for slice_ in self.slices:
            slice_.rot90()
        for roi_coord in self.roi_coordinates:
            roi_coord.rot90()

    def rot270(self):
        """ Inverse of rot90. """
        self.rot90(); self.rot90(); self.rot90()

    def join_with(self, other, xs, ys, smooth_blend=True):
        """ Appends a new ROI to this volume at the given coordinates.

        :param StitchedROI other: The other roi.
        :param float xs, ys: Coordinates of the other ROI.
        :param bool smooth_blend: Whether to taper edges for a smoother blending. It
            assumes other is to the right of self, i.e., |self|other|.
        """
        for slice_, other_slice, x, y in zip(self.slices, other.slices, xs, ys):
            slice_.join_with(other_slice, x, y, smooth_blend)

        # Bookkeeping: move center of other roi to their new values
        for roi_coord in other.roi_coordinates:
            # Add the offsets between center of new_slice and center of old_slice. (seems
            # redundant but necessary when other is already a union of rois)
            roi_coord.xs = [prev_x + (new_x - old_slice.x) for prev_x, new_x, old_slice
                            in zip(roi_coord.xs, xs, other.slices)]
            roi_coord.ys = [prev_y + (new_y - old_slice.y) for prev_y, new_y, old_slice
                            in zip(roi_coord.ys, ys, other.slices)]
            self.roi_coordinates.append(roi_coord)



def linear_stitch(left, right, expected_delta_y, expected_delta_x):
    """ Compute offsets to stitch two side-by-side volumes via cross-correlation.

    Subpixel shifts are calculated per depth and the median is returned.

    Arguments:
    :param left: Slice (height x width) to be stitched in the left.
    :param right: Slice (height x width) to be stitched in the right.
    :param float expected_delta_y: Expected distance between center of left to right.
    :param float expected_delta_x: Expected distance between center of left to right.

    :returns: (delta_y, delta_x). Distance between center of right ROI and left ROI after
        stitching (right_center - left_center)
    """
    # Get some params
    right_height, right_width = right.shape
    left_height, left_width = left.shape
    expected_overlap = left_width / 2 + right_width / 2 - expected_delta_x

    # Drop some rows, columns to avoid artifacts
    skip_columns = int(round(0.05 * expected_overlap)) # 5% of expected overlap
    skip_rows = int(round(0.05 * min(right_height, left_height))) # 5% of top and bottom rows
    left = left[skip_rows: -skip_rows, :-skip_columns]
    right = right[skip_rows: -skip_rows, skip_columns:]
    expected_overlap = int(round(expected_overlap - 2 * skip_columns))

    # Cut strips of expected overlap
    min_height = min(left.shape[0], right.shape[0])
    left_strip = left[:min_height, -expected_overlap:]
    right_strip = right[:min_height, :expected_overlap]

    # Compute best match
    y_shifts, x_shifts = galvo_corrections.compute_motion_shifts(right_strip, left_strip,
                                                                 in_place=False)
    y_shift, x_shift = y_shifts[0], x_shifts[0]

    # Compute right_center minus left_center
    right_ycenter = right_height / 2 - y_shift # original + offset
    right_xcenter = right_width / 2 + (left_width - expected_overlap - 2*skip_columns) - x_shift # original + repositioning + offset
    delta_y = -(right_ycenter - left_height / 2) # negative to change direction of y axis
    delta_x = right_xcenter - left_width / 2

    return delta_y, delta_x