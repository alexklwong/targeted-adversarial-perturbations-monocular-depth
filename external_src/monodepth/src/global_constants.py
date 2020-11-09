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
N_HEIGHT                        = 256
N_WIDTH                         = 512
N_CHANNEL                       = 3

ENCODER_TYPE_AVAILABLE          = ['resnet18',
                                   'resnet34',
                                   'resnet50',
                                   'vggnet08',
                                   'vggnet11',
                                   'vggnet13']
DECODER_TYPE_AVAILABLE          = ['transpose', 'up']
ACTIVATION_FUNC_AVAILABLE       = ['relu', 'leaky_relu', 'elu']
N_PYRAMID                       = 4
ENCODER_TYPE                    = 'vggnet08'
DECODER_TYPE                    = 'up'
ACTIVATION_FUNC                 = 'elu'

# Training settings
N_EPOCH                         = 60
LEARNING_RATES                  = [1.00e-4, 0.50e-4, 0.25e-4, 0.10e-4]
LEARNING_SCHEDULE               = [18, 24]
USE_AUGMENT                     = False
W_COLOR                         = 0.15
W_SSIM                          = 0.85
W_SMOOTHNESS                    = 0.10
W_LEFT_RIGHT                    = 1.00

# Checkpoint settings
N_DISPLAY                       = 4
N_SUMMARY                       = 1000
N_CHECKPOINT                    = 5000
CHECKPOINT_PATH                 = os.path.join('monodepth_models', 'model')

# Depth range settings
SCALE_FACTOR                    = 0.30
MIN_EVALUATE_DEPTH              = 1e-3
MAX_EVALUATE_DEPTH              = 80.0

# Hardware settings
DEVICE                          = 'cuda'
CUDA                            = 'cuda'
CPU                             = 'cpu'
GPU                             = 'gpu'
N_THREAD                        = 8
