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
import os, sys
import torch
sys.path.insert(0, os.path.join(os.getcwd(), 'external_src', 'monodepth2'))
from networks.resnet_encoder import ResnetEncoder
from networks.depth_decoder import DepthDecoder


STEREO_SCALE_FACTOR             = 5.4
MIN_PREDICT_DEPTH               = 0.1
MAX_PREDICT_DEPTH               = 100.0


class Monodepth2Model(object):
    '''

    Wrapper class for Monodepth2

    Args:
        device : torch.device
            device to run on
    '''
    def __init__(self, device=torch.device('cuda')):

        # Depth range settings
        self.scale_factor_depth = STEREO_SCALE_FACTOR
        self.min_predict_depth = MIN_PREDICT_DEPTH
        self.max_predict_depth = MAX_PREDICT_DEPTH

        self.device = device

        # Restore depth prediction network
        self.encoder = ResnetEncoder(18, False)
        self.decoder = DepthDecoder(
            num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

        # Move to device
        self.to(self.device)
        self.encoder.eval()
        self.decoder.eval()

    def forward(self, image):
        '''
        Forwards an image through the network

        Args:
            image : tensor
                N x C x H x W tensor

        Returns:
            tensor : N x C x H x W depth
        '''

        # Forward through network
        disparity_latent = self.encoder(image)
        disparity_outputs = self.decoder(disparity_latent)
        disparity_outputs = disparity_outputs[("disp", 0)]

        # Convert disparity to depth
        min_disparity = 1.0 / self.max_predict_depth
        max_disparity = 1.0 / self.min_predict_depth

        disparity_outputs = \
            min_disparity + (max_disparity - min_disparity) * disparity_outputs

        return self.scale_factor_depth / disparity_outputs

    def to(self, device):
        '''
        Moves model to device

        Args:
            device : torch.device
        '''

        self.encoder.to(device)
        self.decoder.to(device)

    def train(self):
        '''
        Sets model to training mode
        '''

        self.encoder.train()
        self.decoder.train()

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        self.encoder.eval()
        self.decoder.eval()

    def parameters(self):
        '''
        Fetches model parameters

        Returns:
            list : list of model parameters
        '''

        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def restore_model(self, encoder_restore_path, decoder_restore_path):
        '''
        Restores model from checkpoint path

        Args:
            encoder_restore_path : str
                checkpoint for encoder
            decoder_restore_path : str
                checkpoint for decoder
        '''

        # Restore weights for encoder
        loaded_dict_encoder = torch.load(encoder_restore_path, map_location=self.device)

        if 'model_state_dict' in loaded_dict_encoder.keys():
            loaded_dict_encoder = loaded_dict_encoder['model_state_dict']

        filtered_dict_encoder = {
            k : v for k, v in loaded_dict_encoder.items() if k in self.encoder.state_dict()
        }

        self.encoder.load_state_dict(filtered_dict_encoder)

        # Restore weights for decoder
        loaded_dict_decoder = torch.load(decoder_restore_path, map_location=self.device)

        if 'model_state_dict' in loaded_dict_decoder.keys():
            loaded_dict_decoder = loaded_dict_decoder['model_state_dict']

        self.decoder.load_state_dict(loaded_dict_decoder)
