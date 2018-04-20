import numpy as np
from skimage import feature
from scipy import ndimage
import time

from . import enhancement

def register_rigid(stack, field, px_estimate, px_range, angles=(0, 0, 0)):
    """ Registers using skimage.feature match_template.

    Rotates the stack, cross-correlates the field at each position and returns the score,
    coordinates and angles of the best one. We use a right handed coordinate system (x
    points to the right, y towards you, and z downward) with right-handed/clockwise
    rotations.

    Stack is rotated with the inverse of the intrinsic yaw->pitch->roll rotation adding
    some slack above and below the desired slices to avoid black spaces after rotation.
    After rotation, we cut any black spaces in x and y and then run the cross-correlation.

    :param np.array: 3-d stack (depth, height, width).
    :param np.array field: 2-d field to register in the stack.
    :param triplet px_estimate: Initial estimate in x, y, z. (0, 0, 0) in center of stack.
    :param triplet px_range: How many pixels to search around the initial estimate.
    :param triplet angles: Angle in degrees for rotation over z, y and x axis, i.e.
        (yaw, pitch, roll)

    : returns best_score, (x, y, z), (yaw, pitch, roll). Best score is the highest
        correlation found. (x, y, z) are expressed as distances to the center of the
        stack. And yaw, pitch, roll are the same as the input.
    """
    print(time.ctime(), 'Processing angles', angles)

    # Basic checks
    if len(px_estimate) != 3 or len(px_range) != 3 or len(angles) != 3:
        raise ValueError('px_estimate, px_range and angles need to have length 3')
    if np.any(np.array(angles) > 10):
        raise ValueError('register_rigid only works for small angles.')

    # Get rotation matrix
    R = create_rotation_matrix(*angles)
    R_inv = np.linalg.inv(R)

    # Compute the limits of our desired ROI in the rotated stack
    w, h, d = px_range # coordinates of the higher x, y, z in the original stack
    roi_corners = [[-w, -h, -d], [w, -h, -d], [w, h, -d], [-w, h, -d], [-w, -h, d],
                   [w, -h, d], [w, h, d], [-w, h, d]] # clockwise, starting at upper left
    rot_roi_corners = np.dot(R_inv, np.array(roi_corners).T)
    rot_xlim = np.max(rot_roi_corners[0, [1, 2, 5, 6]]) # x that sticks furthest to the right
    rot_ylim = np.max(rot_roi_corners[1, [2, 3, 6, 7]]) # y that sticks furthest down
    rot_zlim = np.max(rot_roi_corners[2, [4, 5, 6, 7]]) # z that sticks furthest away from you
    rot_xlim += field.shape[1] / 2 # add half the field width to still see the full field in the corners
    rot_ylim += field.shape[0] / 2 # add half the field height to still see the full field in the corners

    # Compute how much we can cut of original ROI (for efficiency) but avoiding black spaces
    w, h, d = rot_xlim, rot_ylim, rot_zlim
    rot_big_corners = [[-w, -h, -d], [w, -h, -d], [w, h, -d], [-w, h, -d], [-w, -h, d],
                       [w, -h, d], [w, h, d], [-w, h, d]] # clockwise, starting at upper left
    big_corners = np.dot(R, np.array(rot_big_corners).T)
    x_slack = np.max(big_corners[0, [1, 2, 5, 6]]) # x that sticks furthest to the right
    y_slack = np.max(big_corners[1, [2, 3, 6, 7]]) # y that sticks furthest down
    z_slack = np.max(big_corners[2, [4, 5, 6, 7]]) # z that sticks furthest away from you

    # Cut original stack
    slices = [slice(int(round(max(0, s/2 + p - sl))), int(round(s/2 + p + sl))) for s, p, sl
              in zip(stack.shape, px_estimate[::-1], [z_slack, y_slack, x_slack])]
    mini_stack = stack[slices]

    # Rotate stack (inverse of intrinsic yaw-> pitch -> roll)
    rotated = _inverse_rot3d(mini_stack, *angles) # rotates around center

    # Cut rotated stack (using the limits calculated above)
    px_estimate_ms = [s/2 + p - (sli.start + ms/2) for s, p, sli, ms in # px_estimate with (0, 0, 0) in center of mini_stack
                      zip(stack.shape, px_estimate[::-1], slices, mini_stack.shape)][::-1] # x, y, z
    rot_px_estimate = np.dot(R_inv, px_estimate_ms)
    rot_slices = [slice(int(round(max(0, s/2 + p - sl))), int(round(s/2 + p + sl))) for s, p, sl
                  in zip(rotated.shape, rot_px_estimate[::-1], [rot_zlim, rot_ylim, rot_xlim])]
    mini_rotated = rotated[rot_slices]

    # Create mask to restrict correlations to appropiate range
    mini_mask = np.zeros_like(mini_stack)
    mask_slices = [slice(int(round(max(0, s/2 + p - r))), int(round(s/2 + p + r))) for s, p, r
                   in zip(mini_stack.shape, px_estimate_ms[::-1], px_range[::-1])]
    mini_mask[mask_slices] = 1 # only consider positions initially in the range
    rot_mask = _inverse_rot3d(mini_mask, *angles)
    mr_mask = rot_mask[rot_slices]

    # Crop field FOV to be smaller than the stack's
    cut_rows = max(1, int(np.ceil((field.shape[0] - mini_rotated.shape[1]) / 2)))
    cut_cols = max(1, int(np.ceil((field.shape[1] - mini_rotated.shape[2]) / 2)))
    field = field[cut_rows:-cut_rows, cut_cols:-cut_cols]

    # Sharpen images
    norm_field = enhancement.sharpen_2pimage(field)
    norm_stack = np.stack(enhancement.sharpen_2pimage(s) for s in mini_rotated)

    # 3-d match_template
    corrs = np.stack(feature.match_template(s, norm_field, pad_input=True) for s in norm_stack)
    smooth_corrs = ndimage.gaussian_filter(corrs, 0.7)
    masked_corrs = smooth_corrs * mr_mask
    best_score = np.max(masked_corrs)
    rot_z, rot_y, rot_x = np.unravel_index(np.argmax(masked_corrs), masked_corrs.shape)

    # Express coordinates as distances to rotated center
    rot_xp = rot_slices[2].start + (rot_x + 0.5) - rotated.shape[2] / 2
    rot_yp = rot_slices[1].start + (rot_y + 0.5) - rotated.shape[1] / 2
    rot_zp = rot_slices[0].start + (rot_z + 0.5) - rotated.shape[0] / 2

    # Apply inverse rotation to transform it into distances to mini_stack center
    xp, yp, zp = np.dot(R, (rot_xp, rot_yp, rot_zp))

    # Map back to original stack coordinates
    x_final = (slices[2].start + mini_stack.shape[2] / 2 + xp) - stack.shape[2] / 2 # with respect to original center
    y_final = (slices[1].start + mini_stack.shape[1] / 2 + yp) - stack.shape[1] / 2 # with respect to original center
    z_final = (slices[0].start + mini_stack.shape[0] / 2 + zp) - stack.shape[0] / 2 # with respect to original center

    return best_score, (x_final, y_final, z_final), angles


# TODO: All rotations could be done in a single step using map_coordinates
def _inverse_rot3d(stack, yaw, pitch, roll):
    """ Apply the inverse of an intrinsic yaw -> pitch -> roll rotation.
    inv(yaw(w) -> pitch(v) -> roll(u)) = roll(-u) -> pitch(-v) -> yaw(-w) = extrinsic
        yaw(-w) -> extrinsic pitch(-v) -> extrinsic roll(-u).

    Rotations are applied around the center, thus center of original and rotated stack
    are the same. Shape, however, will not be the same in general, rotated stack could be
    bigger.

    :param np.array stack: 3-d volume to rotate
    :param float yaw: Angle in degrees for rotation over z axis.
    :param float pitch: Angle in degrees for rotation over y axis.
    :param float roll: Angle in degrees for rotation over x axis.

    :returns: 3-d array. Rotated stack.

    ..ref:: danceswithcode.net/engineeringnotes/rotations_in_3d/rotations_in_3d_part1.html
    """
    if yaw == 0 and pitch == 0 and roll == 0:
        rotated = stack.astype(float, copy=True)
    else:
        # Note on ndimage.rotate: Assuming our coordinate system; axes=(1, 2) will do a left
        # handed rotation in z, (0, 2) a right handed rotation in y and (0, 1) a left handed
        # rotation in x.
        rotated = ndimage.rotate(stack, yaw, axes=(1, 2), order=1) # extrinsic yaw(-w)
        rotated = ndimage.rotate(rotated, -pitch, axes=(0, 2), order=1) # extrinsic pitch(-v)
        rotated = ndimage.rotate(rotated, roll, axes=(0, 1), order=1) # extrinsic roll(-u)

    return rotated


def create_rotation_matrix(yaw, pitch, roll):
    """ 3-D rotation matrix to apply a intrinsic yaw-> pitch-> roll rotation.

    We use a right handed coordinate system (x points to the right, y towards you, and z
    downward) with right-handed/clockwise rotations.

    :param float yaw: Angle in degrees for rotation over z axis.
    :param float pitch: Angle in degrees for rotation over y axis.
    :param float roll: Angle in degrees for rotation over x axis.

    :returns: (3, 3) rotation matrix

    ..ref:: danceswithcode.net/engineeringnotes/rotations_in_3d/rotations_in_3d_part1.html
    """
    w, v, u = yaw *np.pi/180, pitch * np.pi/180, roll * np.pi/180 # degrees to radians
    sin, cos = np.sin, np.cos
    rotation_matrix = [
        [cos(v)*cos(w), sin(u)*sin(v)*cos(w) - cos(u)*sin(w), sin(u)*sin(w) + cos(u)*sin(v)*cos(w)],
        [cos(v)*sin(w), cos(u)*cos(w) + sin(u)*sin(v)*sin(w), cos(u)*sin(v)*sin(w) - sin(u)*cos(w)],
        [-sin(v),       sin(u)*cos(v),                        cos(u)*cos(v)]
    ]
    return rotation_matrix


def find_field_in_stack(stack, height, width, x, y, z, yaw=0, pitch=0, roll=0):
    """ Get a cutout of the given height, width dimensions in the rotated stack at x, y, z.

    :param np.array stack: 3-d stack (depth, height, width)
    :param height, width: Height and width of the cutout from the stack.
    :param float x, y, z: Center of field measured as distance from the center in the
        original stack (before rotation).
    :param yaw, pitch, roll: Rotation angles to apply to the field.

    :returns: A height x width np.array at the desired location.
    """
    # Rotate stack (inverse of intrinsic yaw-> pitch -> roll)
    rotated = _inverse_rot3d(stack, yaw, pitch, roll)

    # Compute center of field in the rotated stack
    R_inv = np.linalg.inv(create_rotation_matrix(yaw, pitch, roll))
    rot_center = np.dot(R_inv, [x, y, z])
    center_ind = np.array(rotated.shape) / 2 + rot_center[::-1] # z, y, x

    # Crop field (interpolate at the desired positions)
    z_coords = [center_ind[0] - 0.5] # -0.5 is because our (0.5, 0.5) is np.map_coordinates' (0, 0)
    y_coords = np.arange(height) - height / 2 + center_ind[1] # - 0.5 is added by '- height / 2' term
    x_coords = np.arange(width)  - width / 2 + center_ind[2]
    coords = np.meshgrid(z_coords, y_coords, x_coords)
    out = ndimage.map_coordinates(rotated, [coords[0].reshape(-1), coords[1].reshape(-1),
                                            coords[2].reshape(-1)], order=1)
    field = out.reshape([height, width])

    return field