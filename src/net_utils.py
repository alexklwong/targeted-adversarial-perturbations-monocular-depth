'''
Authors: Alex Wong <alexw@cs.ucla.edu>, Safa Cicek <safacicek@ucla.edu>
If this code is useful to you, please consider citing the following paper:
A. Wong, S. Cicek, and S. Soatto. Targeted Adversarial Perturbations for Monocular Depth Prediction.
https://arxiv.org/pdf/2006.08602.pdf
@inproceedings{wong2020targeted,
    title={Targeted Adversarial Perturbations for Monocular Depth Prediction},
    author={Wong, Alex and Safa Cicek and Soatto, Stefano},
    booktitle={Advances in neural information processing systems},
    year={2020}
}
'''
import torch
import numpy as np
import data_utils


def depth_transform(depths, masks, transform_type=None, value=1.0):
    '''
    Applies transformation to depth maps

    Args:
        depths : tensor
            N x C x H x W depth maps
        masks : tensor
            N x C x H x W binary masks
        transform_type : list[str]
            transformations to apply to depth maps
        values : float
            multiplier/shift constant used for transformation

    Returns:
        tensor : transformed depth map
    '''

    masks_inverse = 1.0 - masks

    if 'flip_horizontal' in transform_type:
        depths = flip_horizontal(depths)

    if 'flip_vertical' in transform_type:
        depths = flip_vertical(depths)

    if 'multiply' in transform_type:
        depths_mask = masks * (value * depths)
        depths_mask_inverse = masks_inverse * depths
        depths = depths_mask + depths_mask_inverse

    if 'translate_horizontal' in transform_type:
        depths = translate_horizontal(depths, value)

    if 'translate_vertical' in transform_type:
        depths = translate_vertical(depths, value)

    if 'remove' in transform_type:
        # Get locations to remove and interpolate
        validity_map = torch.squeeze(1.0 - masks)
        valid_depths = torch.squeeze(validity_map * depths)

        interp_depths = data_utils.interpolate_depth(
            valid_depths.detach().cpu().numpy(),
            validity_map.detach().cpu().numpy())

        depths = torch.from_numpy(interp_depths.astype(np.float32))

        if masks.device.type == 'cuda':
            depths = depths.cuda()

        depths = depths.view(masks.shape)

    if 'move_horizontal' in transform_type:
        # Copy the instance from the depth map and translate
        instances = masks * depths
        move_instances = translate_horizontal(instances, value)
        move_map = translate_horizontal(masks, value)

        # Remove the instance from the scene first
        validity_map = torch.squeeze(1.0 - masks)
        valid_depths = torch.squeeze(validity_map*depths)

        interp_depths = data_utils.interpolate_depth(
            valid_depths.detach().cpu().numpy(),
            validity_map.detach().cpu().numpy())

        depths = torch.from_numpy(interp_depths.astype(np.float32))

        if masks.device.type == 'cuda':
            depths = depths.cuda()

        depths = (1.0 - move_map) * depths + move_instances

    if 'move_vertical' in transform_type:
        # Copy the instance from the depth map and translate
        instances = masks*depths
        move_instances = translate_vertical(instances, value)
        move_map = translate_vertical(masks, value)

        # Remove the instance from the scene first
        validity_map = torch.squeeze(1.0 - masks)
        valid_depths = torch.squeeze(validity_map * depths)

        interp_depths = data_utils.interpolate_depth(
            valid_depths.detach().cpu().numpy(),
            validity_map.detach().cpu().numpy())

        depths = torch.from_numpy(interp_depths.astype(np.float32))

        if masks.device.type == 'cuda':
            depths = depths.cuda()

        depths = (1.0 - move_map) * depths + move_instances

    return depths

def flip_horizontal(T):
    '''
    Horizontally flips a tensor

    Args:
        T : tensor
            N x C x H x W tensor

    Returns:
        tensor : horizontally flipped tensor
    '''

    return torch.flip(T, dims=[-1])

def flip_vertical(T):
    '''
    Vertically flips a tensor

    Args:
        T : tensor
            N x C x H x W tensor

    Returns:
        tensor : vertically flipped tensor
    '''

    return torch.flip(T, dims=[-2])

def translate_horizontal(T, d):
    '''
    Horizontally translates a tensor

    Args:
        T : tensor
            N x C x H x W tensor
        d : int
            amount to horizontally translate

    Returns:
        tensor : N x C x H x W horizontally translated tensor
    '''

    d = int(d)
    n_width = list(T.shape)[3]

    # Translate towards the right
    if d > 0:
        pad = torch.unsqueeze(T[:, :, :, 0], dim=-1)
        pad = pad.repeat(1, 1, 1, d)
        T = torch.cat([pad, T], dim=-1)
        T = T[:, :, :, 0:n_width]
    # Translate towards the left
    elif d < 0:
        pad = torch.unsqueeze(T[:, :, :, n_width-1], dim=-1)
        pad = pad.repeat(1, 1, 1, abs(d))
        T = torch.cat([T, pad], dim=-1)
        # Select from d:n_width+d
        T = T[:, :, :, -n_width:]

    return T

def translate_vertical(T, d):
    '''
    Vertically translates a tensor

    Args:
        T : tensor
            N x C x H x W tensor
        d : int
            amount to vertically translate

    Returns:
        tensor : N x C x H x W vertically translated tensor
    '''

    d = int(d)
    n_height = list(T.shape)[2]

    # Translate up
    if d > 0:
        pad = torch.unsqueeze(T[:, :, n_height-1, :], dim=-2)
        pad = pad.repeat(1, 1, d, 1)
        T = torch.cat([T, pad], dim=-2)
        # Selects from d:n_height+d
        T = T[:, :, -n_height:, :]
    # Translate down
    elif d < 0:
        pad = torch.unsqueeze(T[:, :, 0, :], dim=-2)
        pad = pad.repeat(1, 1, abs(d), 1)
        T = torch.cat([pad, T], dim=-2)
        T = T[:, :, 0:n_height, :]

    return T
