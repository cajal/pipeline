""" Code to perform  3d stack segmentations. """
from bl3d import models, utils
import torch
import numpy as np
from os import path


#torch.backends.cudnn.benchmark=True # time efficient but memory inefficient


def segment(stack, method='ensemble', pad_mode='reflect', seg_threshold=0.8,
            min_voxels=65, max_voxels=4168, compactness_factor=0.05):
    """ Utility function to segment a 3-d stack

    :param stack: 3-d array. Raw Stack resampled to 1 mm^3.
    :param method: A string from 'single' or 'ensemble'. Whether to use a single model or
        an ensemble of models for segmentation.
    :param pad_mode: How will the stack be padded. Any valid mode from np.pad.
    :param seg_threshold: Threshold used to produce instance segmentations.
    :param min_voxels: Minimum number of voxels in a valid object.
    :param max_voxels: Maximum number of voxels in a valid object.
    :param compactness_factor: Weight for the compactness objective during instance
        segmentation.

    :return: detection, segmentation, instance. Arrays of the same shape as stack:
        voxel-wise centroid probability (np.float32), voxel-wise cell probability
        (np.float32) and instance segmentation (np.int32).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print('Running 3-d segmentation in the CPU will take more time.')

    # Prepare input
    padded = np.pad(stack, 20, mode=pad_mode)
    lcned= utils.lcn(padded, (3, 25, 25))
    norm = (lcned - lcned.mean()) / lcned.std()
    input_ = torch.as_tensor(norm[np.newaxis, np.newaxis, ...])  # 1 x 1 x D x H x W
    del padded, lcned, norm # release memory

    # Declare models
    net = models.QCANet()
    data_path = '/data/pipeline/python/pipeline/data/'
    if method == 'single':
        model_names = ['bestndn_1-9-17026.pth']
    else:
        model_names = ['bestndn_1-9-17026.pth', 'bestndn_1-17-17206.pth',
                       'bestndn_1-3-17259.pth', 'bestndn_1-8-17261.pth']  # we'll ensemble all of these

    # Create detection and segmentation probabilities
    detection_sum = np.empty(input_.shape[-3:], dtype=np.float32)
    segmentation_sum = np.empty(input_.shape[-3:], dtype=np.float32)
    with torch.no_grad():
        for model_name in model_names:
            # Import model from file
            net.load_state_dict(torch.load(path.join(data_path, model_name)))
            net.eval()
            net.to(device)

            # Segment
            detection, segmentation = net.forward_on_big_input(input_)
            detection_sum += torch.sigmoid(detection).squeeze().numpy()
            segmentation_sum += torch.sigmoid(segmentation).squeeze().numpy()
        detection = detection_sum / len(model_names)
        segmentation = segmentation_sum / len(model_names)
    del input_, detection_sum, segmentation_sum # release memory

    # Drop padding (added above)
    detection = detection[20:-20, 20:-20, 20:-20]
    segmentation = segmentation[20:-20, 20:-20, 20:-20]

    # Create instance segmentation
    instance = utils.prob2labels(detection, segmentation, seg_threshold=seg_threshold,
                                 min_voxels=min_voxels, max_voxels=max_voxels,
                                 compactness_factor=compactness_factor)

    return detection, segmentation, instance