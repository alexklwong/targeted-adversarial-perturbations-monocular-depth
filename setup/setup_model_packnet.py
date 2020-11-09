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


PACKNET_VELSUP_CSK_MODEL_URL = 'https://drive.google.com/uc?id=1PLeCOMZjki6XSJmGPOF2Tc2iRvKMIBN4'

PACKNET_MODELS_DIRPATH = os.path.join('pretrained_models', 'packnet')
PACKNET_VELSUP_CSK_MODEL_FILENAME = 'packnet_velsup_csk.zip'
PACKNET_VELSUP_CSK_MODEL_FILEPATH = os.path.join(
    PACKNET_MODELS_DIRPATH, PACKNET_VELSUP_CSK_MODEL_FILENAME)

if not os.path.exists(PACKNET_MODELS_DIRPATH):
    os.makedirs(PACKNET_MODELS_DIRPATH)

if not os.path.exists(PACKNET_VELSUP_CSK_MODEL_FILEPATH):
    print('Downloading packnet velocity Cityscape/KITTI model to {}'.format(
        PACKNET_VELSUP_CSK_MODEL_FILEPATH))
    gdown.download(PACKNET_VELSUP_CSK_MODEL_URL, PACKNET_VELSUP_CSK_MODEL_FILEPATH, quiet=False)
else:
    print('Found packnet velocity Cityscape/KITTI model in {}'.format(
        PACKNET_VELSUP_CSK_MODEL_FILEPATH))

with ZipFile(PACKNET_VELSUP_CSK_MODEL_FILEPATH, 'r') as zip_file:
    zip_file.extractall(PACKNET_MODELS_DIRPATH)
