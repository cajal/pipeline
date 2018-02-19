import numpy as np
from skimage import feature
from scipy import ndimage

import time

def register_rigid(stack, field, z_estimate, z_range, yaw=0, pitch=0, roll=0):
    """ Registers using skimage.feature match_template.

    Rotates the stack, cross-correlates the field at each position and returns the score,
    coordinates and angles of the best one. We use a right handed coordinate system (x
    points to the right, y towards you, and z downward) with right-handed/clockwise
    rotations.

    Stack is rotated with the inverse of the intrinsic yaw->pitch->roll rotation adding
    some slack above and below the desired slices to avoid black spaces after rotation.
    After rotation, we cut any black spaces in x and y and then run the cross-correlation.

    Note on ndimage.rotate: Assuming our coordinate system; axes=(1, 2) will do a left
    handed rotation in z, (0, 2) a right handed rotation in y and (0, 1) a left handed
    rotation in x.

    :param np.array: 3-d stack (depth, height, width).
    :param np.array field: 2-d field to register in the stack.
    :param float z_estimate: Initial estimate of best z.
    :param float z_range: How many slices to search above (and below) z_estimate.
    :param float yaw: Angle in degrees for rotation over z axis.
    :param float pitch: Angle in degrees for rotation over y axis.
    :param float roll: Angle in degrees for rotation over x axis.

    : returns best_score, (x, y, z), (yaw, pitch, roll). Best score is the highest
        correlation found. (x, y, z) are expressed as distances to the center of the
        stack. And yaw, pitch, roll are the same as the input.
    """
    print(time.ctime(), 'Processing angles', (yaw, pitch, roll))

    # Get rotation matrix
    R = create_rotation_matrix(yaw, pitch, roll)
    R_inv = np.linalg.inv(R)

    # Compute how high in z we'll need to cut the rotated stack to account for z = z_range
    h, w = stack.shape[1] / 2, stack.shape[2] / 2 # y, x
    corners_at_zrange = [[-w, -h, z_range], [w, -h, z_range], [w, h, z_range], [-w, h, z_range]] # clockwise, starting at upper left
    rot_corners_at_zrange = np.dot(R_inv, np.array(corners_at_zrange).T)
    rot_ztop = np.max(rot_corners_at_zrange[2])

    # Calculate amount of slack in the initial stack needed to avoid black voxels after rotation.
    corners_at_zero = [[-w, -h, 0], [w, -h, 0], [w, h, 0], [-w, h, 0]]
    corners_at_100= [[-w, -h, 100], [w, -h, 100], [w, h, 100], [-w, h, 100]]
    rot_corners_at_zero = np.dot(R_inv, np.array(corners_at_zero).T) #
    rot_corners_at_100 = np.dot(R_inv, np.array(corners_at_100).T)
    rot_corners_at_ztop = find_intersecting_point(rot_corners_at_zero, rot_corners_at_100, rot_ztop)
    corners_at_ztop = np.dot(R, rot_corners_at_ztop) # map back to original stack coordinates
    z_slack = np.max(corners_at_ztop[2])

    # Restrict stack to relevant part in z
    mini_stack = stack[max(0, int(round(z_estimate - z_slack))): int(round(z_estimate + z_slack))]

    # Rotate stack (inverse of intrinsic yaw-> pitch -> roll)
    rotated = ndimage.rotate(mini_stack, yaw, axes=(1, 2), order=1) # yaw(-w)
    rotated = ndimage.rotate(rotated, -pitch, axes=(0, 2), order=1) # pitch(-v)
    rotated = ndimage.rotate(rotated, roll, axes=(0, 1), order=1) # roll(-u)

    # Calculate where to cut rotated stack (at z_est)
    z_est = (z_estimate - max(0, int(round(z_estimate - z_slack)))) - mini_stack.shape[0] / 2
    rot_z_est = np.dot(R_inv, [0, 0, z_est])[2]
    min_z, max_z = rot_z_est - rot_ztop, rot_z_est + rot_ztop
    top_corners = find_intersecting_point(rot_corners_at_zero, rot_corners_at_100, max(-rotated.shape[0] / 2, min_z))
    bottom_corners = find_intersecting_point(rot_corners_at_zero, rot_corners_at_100, min(rotated.shape[0] / 2, max_z))
    min_x = max(*top_corners[0, [0, 3]], *bottom_corners[0, [0, 3]])
    max_x = min(*top_corners[0, [1, 2]], *bottom_corners[0, [1, 2]])
    min_y = max(*top_corners[1, [0, 1]], *bottom_corners[1, [0, 1]])
    max_y = min(*top_corners[1, [2, 3]], *bottom_corners[1, [2, 3]])

    # Cut rotated stack
    mini_rotated = rotated[max(0, int(round(rotated.shape[0] / 2 + min_z))): int(round(rotated.shape[0] / 2 + max_z)),
                           max(0, int(round(rotated.shape[1] / 2 + min_y))): int(round(rotated.shape[1] / 2 + max_y)),
                           max(0, int(round(rotated.shape[2] / 2 + min_x))): int(round(rotated.shape[2] / 2 + max_x))]
    z_center = rotated.shape[0] / 2 - max(0, int(round(rotated.shape[0] / 2 + min_z)))
    y_center = rotated.shape[1] / 2 - max(0, int(round(rotated.shape[1] / 2 + min_y)))
    x_center = rotated.shape[2] / 2 - max(0, int(round(rotated.shape[2] / 2 + min_x)))

    # Crop field FOV to be smaller than the stack's
    cut_rows = max(1, int(np.ceil((field.shape[0] - mini_rotated.shape[1]) / 2)))
    cut_cols = max(1, int(np.ceil((field.shape[1] - mini_rotated.shape[2]) / 2)))
    field = field[cut_rows:-cut_rows, cut_cols:-cut_cols]

    # 3-d match_template
    corrs = np.stack(feature.match_template(s, field, pad_input=True) for s in mini_rotated)
    smooth_corrs = ndimage.gaussian_filter(corrs, 0.7)
    best_score = np.max(smooth_corrs)
    z, y, x = np.unravel_index(np.argmax(smooth_corrs), smooth_corrs.shape)

    # Express coordinates as distances to mini_stack/rotated center
    x_offset = (x + 0.5) - x_center
    y_offset = (y + 0.5) - y_center
    z_offset = (z + 0.5) - z_center

    # Map back to original stack coordinates
    xp, yp, zp = np.dot(R, [x_offset, y_offset, z_offset])
    zp = ((max(0, int(round(z_estimate - z_slack))) + mini_stack.shape[0] / 2 + zp) - stack.shape[0] / 2) # with respect to original z center

    return best_score, (xp, yp, zp), (yaw, pitch, roll)


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


def find_intersecting_point(p1, p2, z):
    """ Find a point at a given z in the line that crosses p1 and p2.

    :param np.array p1: A point (1-d array of size 3) or a matrix of points (3 x n).
    :param np.array p2: A point (1-d array of size 3) or a matrix of points (3 x n).
        Number of points in p2 needs to match p1.
    :param float z: z to insersect the line.

    :returns: A point (1-d array of size 3) or a matrix of points (3 x n).

    ..ref:: https://brilliant.org/wiki/3d-coordinate-geometry-equation-of-a-line/
    """
    direction_vector = p2 - p1
    if any(abs(direction_vector[2]) < 1e-10):
        raise ArithmeticError('Line is parallel to the z-plane. Infinite or no solutions.')
    q = p1 + ((z - p1[2]) / direction_vector[2]) * direction_vector # p1 + ((z-z1)/ n) * d
    return q