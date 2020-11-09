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


def color_consistency_loss_func(src, tgt):
    '''
    Computes color consistency loss

    Args:
        src : tensor
            N x C x H x W source array
        tgt : tensor
            N x C x H x W target array

    Returns:
        float : color consistency loss
    '''

    return torch.mean(torch.abs(tgt - src))

def ssim_loss_func(src, tgt):
    '''
    Computes SSIM loss

    Args:
        src : numpy
            N x C x H x W source array
        tgt : numpy
            N x C x H x W target array

    Returns:
        float : SSIM loss
    '''

    return torch.mean(ssim(src, tgt))

def smoothness_loss_func(predict, image):
    '''
    Computes local smoothness loss

    Args:
        predict : numpy
            N x 1 x H x W source array
        image : numpy
            N x C x H x W image

    Returns:
        float : smoothness loss
    '''

    predict_dy, predict_dx = gradient_yx(predict)
    image_dy, image_dx = gradient_yx(image)

    weights_x = torch.exp(-torch.mean(torch.abs(image_dx), dim=1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_dy), dim=1, keepdim=True))

    smoothness_x = torch.mean(weights_x * torch.abs(predict_dx))
    smoothness_y = torch.mean(weights_y * torch.abs(predict_dy))

    return smoothness_x + smoothness_y

def left_right_consistency_loss_func(src, tgt):
    '''
    Computes left-right consistency loss

    Args:
        src : numpy
            N x C x H x W source array
        tgt : numpy
            N x C x H x W target array

    Returns:
        float : left-right consistency loss
    '''

    return torch.mean(torch.abs(src - tgt))

def gradient_yx(T):
    '''
    Computes y-x gradients of a tensor

    Args:
        T : tensor
            N x C x H x W tensor

    Returns:
        tensor : N x C x H-1 x W gradients in y-direction
        tensor : N x C x H x W-1 gradients in x-direction
    '''

    dx = T[:, :, :, :-1]-T[:, :, :, 1:]
    dy = T[:, :, :-1, :]-T[:, :, 1:, :]
    return dy, dx

def ssim(x, y):
    '''
    Computes SSIM between two tensors

    Args:
        x : tensor
            N x C x H x W source tensor
        y : tensor
            N x C x H x W target tensor

    Returns:
        float : SSIM score
    '''

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = torch.nn.AvgPool2d(3, 1)(x)
    mu_y = torch.nn.AvgPool2d(3, 1)(y)
    mu_xy = mu_x * mu_y
    mu_xx = mu_x ** 2
    mu_yy = mu_y ** 2

    sigma_x = torch.nn.AvgPool2d(3, 1)(x ** 2) - mu_xx
    sigma_y = torch.nn.AvgPool2d(3, 1)(y ** 2) - mu_yy
    sigma_xy = torch.nn.AvgPool2d(3, 1)(x * y) - mu_xy

    numer = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denom = (mu_xx + mu_yy + C1) * (sigma_x+sigma_y + C2)
    score = numer / denom

    return torch.clamp((1.0 - score) / 2.0, 0.0, 1.0)
