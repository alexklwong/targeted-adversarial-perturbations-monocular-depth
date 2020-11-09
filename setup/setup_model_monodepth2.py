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


MONODEPTH2_STEREO_MODEL_URL = 'https://drive.google.com/uc?id=1ArW1Tr9-Clukepy0_olWw8AHMbigOXTH'

MONODEPTH2_MODELS_DIRPATH = os.path.join('pretrained_models', 'monodepth2')
MONODEPTH2_STEREO_MODEL_FILENAME = 'stereo_640x192.zip'
MONODEPTH2_STEREO_MODEL_FILEPATH = os.path.join(
    MONODEPTH2_MODELS_DIRPATH, MONODEPTH2_STEREO_MODEL_FILENAME)

if not os.path.exists(MONODEPTH2_MODELS_DIRPATH):
    os.makedirs(MONODEPTH2_MODELS_DIRPATH)

if not os.path.exists(MONODEPTH2_STEREO_MODEL_FILEPATH):
    print('Downloading monodepth2 stereo 640x192 model to {}'.format(
        MONODEPTH2_STEREO_MODEL_FILEPATH))
    gdown.download(MONODEPTH2_STEREO_MODEL_URL, MONODEPTH2_STEREO_MODEL_FILEPATH, quiet=False)
else:
    print('Found monodepth2 stereo 640x192 model in {}'.format(
        MONODEPTH2_STEREO_MODEL_FILEPATH))

with ZipFile(MONODEPTH2_STEREO_MODEL_FILEPATH, 'r') as zip_file:
    zip_file.extractall(MONODEPTH2_MODELS_DIRPATH)
