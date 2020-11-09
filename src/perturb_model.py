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
import torch
import global_constants as settings


EPSILON = 1e-8


class Perturbations(torch.nn.Module):
    '''
    Base perturbations image class

    Args:
        n_height : int
            height of perturbations image
        n_width : int
            width of perturbations image
        n_channel : int
            number of channels of perturbations image
        output_norm : float
            upper norm of perturbations
    '''

    def __init__(self,
                 n_height=settings.N_HEIGHT,
                 n_width=settings.N_WIDTH,
                 n_channel=settings.N_CHANNEL,
                 output_norm=settings.OUTPUT_NORM):
        super(Perturbations, self).__init__()

        self.output_norm = output_norm

        # Initialize noise image
        self.weights = torch.nn.Parameter(
            torch.zeros([n_channel, n_height, n_width],
            requires_grad=True))

    def forward(self, x):
        # Clip the perturbations outside of upper norm
        noise = torch.clamp(self.weights, -self.output_norm, self.output_norm)
        return x + noise


class PerturbationsModel(object):
    '''
    Perturbations model class

    Args:
        n_height : int
            height of perturbations image
        n_width : int
            width of perturbations image
        n_channel : int
            number of channels of perturbations image
        output_norm : float
            upper norm of perturbations
        device : torch.device
            device to run on
    '''

    def __init__(self,
                 n_height=settings.N_HEIGHT,
                 n_width=settings.N_WIDTH,
                 n_channel=settings.N_CHANNEL,
                 output_norm=settings.OUTPUT_NORM,
                 device=torch.device(settings.CUDA)):

        self.device = device

        self.perturbations = Perturbations(
            n_height=n_height,
            n_width=n_width,
            n_channel=n_channel,
            output_norm=output_norm)

        # Move to device
        self.to(self.device)

    def forward(self, image):
        '''
        Applies perturbations to image

        Args:
            image : tensor
                N x C x H x W tensor

        Returns:
            tensor : N x C x H x W perturbations
            tensor : N x C x H x W perturbed image
        '''

        # Apply perturbations to image
        outputs = self.perturbations(image)

        # Clip image between 0.0 and 1.0 and recover the noise
        image_output = torch.clamp(outputs, 0.0, 1.0)
        noise_output = image_output - image

        return noise_output, image_output

    def compute_loss(self, depth_output, depth_target):
        '''
        Computes target consistency loss

        Args:
            depth_output : tensor
                N x 1 x H x W source tensor
            depth_target : tensor
                N x 1 x H x W target tensor

        Returns:
            float : loss
        '''

        # Compute target depth consistency loss function
        delta = torch.abs(depth_target - depth_output)
        delta = delta / (depth_target + EPSILON)
        loss = torch.mean(delta)

        return loss

    def to(self, device):
        '''
        Moves model to device

        Args:
            device : torch.device
        '''

        self.perturbations.to(device)

    def train(self):
        '''
        Sets model to training mode
        '''

        self.perturbations.train()

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        self.perturbations.eval()

    def parameters(self):
        '''
        Fetches model parameters

        Returns:
            list : list of model parameters
        '''

        return list(self.perturbations.parameters())
