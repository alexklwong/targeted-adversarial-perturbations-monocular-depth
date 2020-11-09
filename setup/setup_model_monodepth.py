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
import os, gdown
from zipfile import ZipFile


MONODEPTH_RESNET50_MODEL_URL = 'https://drive.google.com/uc?id=1yJ8wx9khv-pg-D_UQIrvAAWW3zsM3OGA'

MONODEPTH_MODELS_DIRPATH = os.path.join('pretrained_models', 'monodepth')
MONODEPTH_RESNET50_MODEL_FILENAME = 'resnet50.zip'
MONODEPTH_RESNET50_MODEL_FILEPATH = os.path.join(
    MONODEPTH_MODELS_DIRPATH, MONODEPTH_RESNET50_MODEL_FILENAME)

if not os.path.exists(MONODEPTH_MODELS_DIRPATH):
    os.makedirs(MONODEPTH_MODELS_DIRPATH)

if not os.path.exists(MONODEPTH_RESNET50_MODEL_FILEPATH):
    print('Downloading monodepth ResNet50 model to {}'.format(
        MONODEPTH_RESNET50_MODEL_FILEPATH))
    gdown.download(MONODEPTH_RESNET50_MODEL_URL, MONODEPTH_RESNET50_MODEL_FILEPATH, quiet=False)
else:
    print('Found monodepth ResNet50 model in {}'.format(
        MONODEPTH_RESNET50_MODEL_FILEPATH))

with ZipFile(MONODEPTH_RESNET50_MODEL_FILEPATH, 'r') as zip_file:
    zip_file.extractall(MONODEPTH_MODELS_DIRPATH)
