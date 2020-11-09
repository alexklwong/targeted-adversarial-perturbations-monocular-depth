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
import numpy as np


def root_mean_sq_err(src, tgt):
    '''
    Computes the absolute relative error metric

    Args:
        src : numpy
            source array
        tgt : numpy
            target array

    Returns:
        float : absolute relative error
    '''

    return np.sqrt(np.mean((src - tgt) ** 2))

def mean_abs_err(src, tgt):
    '''
    Computes the mean absolute error metric

    Args:
        src : numpy
            source array
        tgt : numpy
            target array

    Returns:
        float : mean absolute error
    '''

    return np.mean(np.abs(src - tgt))

def log_root_mean_sq_err(src, tgt):
    '''
    Computes the log root mean squared error metric

    Args:
        src : numpy
            source array
        tgt : numpy
            target array

    Returns:
        float : log root mean squared error
    '''

    return np.sqrt(np.mean((np.log(src) - np.log(tgt)) ** 2))

def abs_rel_err(src, tgt):
    '''
    Computes the absolute relative error metric

    Args:
        src : numpy
            source array
        tgt : numpy
            target array

    Returns:
        float : absolute relative error
    '''

    return np.mean(np.abs(src - tgt) / tgt)

def sq_rel_err(src, tgt):
    '''
    Computes the squared relative error metric

    Args:
        src : numpy
            source array
        tgt : numpy
            target array

    Returns:
        float : squared relative error
    '''

    return np.mean(((src - tgt) ** 2) / tgt)

def thresh_ratio_err(src, tgt, thresh=1.25):
    '''
    Computes the thresholded ratio (a1) error metric

    Args:
        src : numpy
            source array
        tgt : numpy
            target array

    Returns:
        float : thresholded ratio (a1) error
    '''

    ratio = np.maximum((tgt / src), (src / tgt))
    return np.mean(ratio < thresh)
