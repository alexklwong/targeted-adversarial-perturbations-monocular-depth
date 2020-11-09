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
import os


# Batch settings
N_BATCH                         = 8
N_HEIGHT                        = 192
N_WIDTH                         = 640
N_CHANNEL                       = 3

# Run settings
DEPTH_METHOD_AVAILABLE          = ['monodepth',
                                   'monodepth2',
                                   'packnet']
DEPTH_TRANSFORM_FUNC_AVAILABLE  = ['none',
                                   'multiply',
                                   'flip_vertical',
                                   'flip_horizontal',
                                   'translate_vertical',
                                   'translate_horizontal',
                                   'remove',
                                   'move_horizontal',
                                   'move_vertical']
MASK_CONSTRAINT_AVAILABLE       = ['within_mask',
                                   'out_of_mask',
                                   'none']

N_STEP                          = 200
OUTPUT_NORM                     = 1.00
LEARNING_RATES                  = [2.00]
LEARNING_SCHEDULE               = [600]
DEPTH_METHOD                    = 'monodepth2'
DEPTH_TRANSFORM_FUNC            = 'none'
DEPTH_TRANSFORM_VALUE           = 1.00
MASK_CONSTRAINT                 = 'none'

# Checkpoint settings
CHECKPOINT_PATH                 = os.path.join('perturbations', 'monodepth2')
DEPTH_MODEL_RESTORE_PATH0       = ''
DEPTH_MODEL_RESTORE_PATH1       = ''

# Hardware settings
DEVICE                          = 'cuda'
CUDA                            = 'cuda'
CPU                             = 'cpu'
GPU                             = 'gpu'
N_THREAD                        = 8
