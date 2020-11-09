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
sys.path.insert(0, os.path.join(os.getcwd(), 'external_src', 'packnet'))
from packnet_sfm.models.SfmModel import SfmModel
from packnet_sfm.networks.depth.PackNet01 import PackNet01
from packnet_sfm.networks.pose.PoseNet import PoseNet


class PackNetModel(object):
    '''
    Wrapper class for PackNet

    Args:
        device : torch.device
            device to run on
    '''
    def __init__(self, device=torch.device('cuda')):

        self.device = device

        self.model = SfmModel()
        self.model.depth_net = PackNet01()
        self.model.pose_net = PoseNet()

        # Move to device
        self.to(self.device)

        self.model.eval()
        self.model.depth_net.eval()
        self.model.pose_net.eval()

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
        disparity_outputs = self.model.compute_inv_depths(image)
        depth_outputs = 1.0 / disparity_outputs[-1]

        return depth_outputs

    def to(self, device):
        '''
        Moves model to device

        Args:
            device : torch.device
        '''

        self.model.to(device)
        self.model.depth_net.to(device)
        self.model.pose_net.to(device)

    def train(self):
        '''
        Sets model to training mode
        '''

        self.model.eval()
        self.model.depth_net.eval()
        self.model.pose_net.eval()

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        self.model.eval()
        self.model.depth_net.eval()
        self.model.pose_net.eval()

    def parameters(self):
        '''
        Fetches model parameters

        Returns:
            list : list of model parameters
        '''

        return self.model.parameters()

    def restore_model(self, restore_path):
        '''
        Restores model from checkpoint path

        Args:
            restore_path : str
                checkpoint for model
        '''

        # Restore weights for encoder
        checkpoint = torch.load(restore_path, map_location=self.device)
        state_dict = checkpoint['state_dict']
        state_dict_clean = {}

        for name in state_dict:
            key = name.split('.', 1)[1]
            state_dict_clean[key] = state_dict[name]

        self.model.load_state_dict(state_dict_clean)
