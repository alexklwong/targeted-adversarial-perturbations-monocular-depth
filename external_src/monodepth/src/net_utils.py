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


def warp1d_horizontal(image, disparity):
    '''
    Performs horizontal warping

    Args:
        image : tensor
            N x C x H x W image
        disparity : tensor
            N x 1 x H x W disparity
    '''
    n_batch, _, n_height, n_width = image.shape

    # Original coordinates of pixels
    x = torch.linspace(0, 1, n_width, dtype=torch.float32, device=image.device) \
        .repeat(n_batch, n_height, 1)
    y = torch.linspace(0, 1, n_height, dtype=torch.float32, device=image.device) \
        .repeat(n_batch, n_width, 1) \
        .transpose(1, 2)

    # Apply shift in X direction
    dx = disparity[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
    flow_field = torch.stack((x+dx, y), dim=3)

    # In grid_sample coordinates are assumed to be between -1 and 1
    return torch.nn.functional.grid_sample(image,
        grid=(2 * flow_field - 1),
        mode='bilinear',
        padding_mode='zeros')
