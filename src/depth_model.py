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
sys.path.insert(0, os.path.join(os.getcwd(), 'external_src', 'monodepth', 'src'))


class DepthModel(object):
    '''
    Wrapper class for monocular depth models

    Args:
        method : str
            monocular depth method to use
        device : torch.device
            device to run on
    '''
    def __init__(self,
                 method='monodepth2',
                 device=torch.device('cuda')):

        self.method = method

        if method == 'monodepth':
            from monodepth_model import MonodepthModel
            self.model = MonodepthModel(
                encoder_type='resnet50',
                decoder_type='up',
                activation_func='elu',
                n_pyramid=4,
                scale_factor=0.30,
                device=device)
        elif method == 'monodepth2':
            from monodepth2_model import Monodepth2Model
            self.model = Monodepth2Model(device=device)
        elif method == 'packnet':
            from packnet_model import PackNetModel
            self.model = PackNetModel(device=device)

    def forward(self, image, camera):
        '''
        Forwards an image through the network

        Args:
            image : tensor
                N x C x H x W tensor
            camera : tensor
                N x 3 x 3 intrinsics

        Returns:
            tensor : N x C x H x W depth
        '''

        self.image = image

        if self.method == 'monodepth':
            _, _, depth, _ = self.model.forward(image, camera)
        elif self.method == 'monodepth2':
            depth = self.model.forward(image)
        elif self.method == 'packnet':
            depth = self.model.forward(image)

        return depth

    def to(self, device):
        '''
        Moves model to device

        Args:
            device : torch.device
        '''

        self.model.to(device)

    def train(self):
        '''
        Sets model to training mode
        '''

        self.model.train()

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        self.model.eval()

    def parameters(self):
        '''
        Fetches model parameters

        Returns:
            list : list of model parameters
        '''

        return self.model.parameters()

    def restore_model(self, restore_path0, restore_path1=None):
        '''
        Restores model from checkpoint path

        Args:
            restore_path0 : str
                checkpoint for entire model or encoder
            restore_path1 : str
                checkpoint for decoder
        '''

        if self.method == 'monodepth':
            self.model.restore_model(
                encoder_restore_path=restore_path0,
                decoder_restore_path=restore_path1)
        elif self.method == 'monodepth2':
            self.model.restore_model(
                encoder_restore_path=restore_path0,
                decoder_restore_path=restore_path1)
        elif self.method == 'packnet':
            self.model.restore_model(restore_path=restore_path0)
