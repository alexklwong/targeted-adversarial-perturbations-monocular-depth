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

def lp_norm(T, p=1.0, axis=None):
    '''
    Computes the Lp norm of a tensor

    Args:
        T : tensor
            H x W x C tensor
        p : int
            norm degree
        axis : int
            axis to compute Lp norm

    Returns:
        int : Lp norm
    '''

    if p != 0 and axis is None:
        return np.mean(np.abs(T))
    else:
        if p != 0:
            return np.mean(np.sum(np.abs(T)**p, axis=axis)**(1.0/p))
        else:
            return np.max(np.abs(T))
