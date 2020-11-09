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
import loss_utils, net_utils
import networks


class MonodepthModel(object):
    '''
    Monodepth model

    Args:
        encoder_type : str
            encoder type to use: resnet or vggnet
        decoder_type : str
            tranpose or up convolution
        activation_func : str
            activation function to use
        n_pyramid : int
            depth of loss pyramid
        scale_factor : int
            scaling factor of disparity
        device : torch.device
            device to run on
    '''
    def __init__(self,
                 encoder_type='resnet50',
                 decoder_type='up',
                 activation_func='elu',
                 n_pyramid=4,
                 scale_factor=0.30,
                 device=torch.device('cuda')):

        self.scale_factor = scale_factor

        # Network architecture
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.n_pyramid = n_pyramid
        self.device = device

        # Select activation function
        if activation_func == 'relu':
            activation_func = torch.nn.ReLU()
        elif activation_func == 'leaky_relu':
            activation_func = torch.nn.LeakyReLU(negative_slope=0.10, inplace=True)
        elif activation_func == 'elu':
            activation_func = torch.nn.ELU()
        else:
            activation_func = torch.nn.ELU()

        # Build network
        n_filters_encoder = [32, 64, 128, 256, 256]
        n_filters_decoder = [256, 128, 64, 32, 16]
        n_skips_decoder = [256, 128, 64, 32, 0]
        if self.encoder_type == 'resnet18':
            self.encoder = networks.ResNetEncoder(
                n_layer=18,
                n_filters=n_filters_encoder,
                activation_func=activation_func)
        elif self.encoder_type == 'resnet34':
            self.encoder = networks.ResNetEncoder(
                n_layer=34,
                n_filters=n_filters_encoder,
                activation_func=activation_func)
        elif self.encoder_type == 'resnet50':
            self.encoder = networks.ResNetEncoder(
                n_layer=50,
                n_filters=n_filters_encoder,
                activation_func=activation_func)
            # ResNet bottleneck uses 1x1 kernel with 4x filters
            n_filters_encoder[-1] = 4*n_filters_encoder[-1]
            n_skips_decoder = [4*n for n in n_skips_decoder[:-2]]+n_skips_decoder[-2:]
        elif self.encoder_type == 'vggnet08':
            self.encoder = networks.VGGNetEncoder(
                n_layer=8,
                n_filters=n_filters_encoder,
                activation_func=activation_func)
        elif self.encoder_type == 'vggnet11':
            self.encoder = networks.VGGNetEncoder(
                n_layer=11,
                n_filters=n_filters_encoder,
                activation_func=activation_func)
        elif self.encoder_type == 'vggnet13':
            self.encoder = networks.VGGNetEncoder(
                n_layer=13,
                n_filters=n_filters_encoder,
                activation_func=activation_func)

        self.decoder = networks.DisparityDecoder(
            input_channels=n_filters_encoder[-1],
            n_filters=n_filters_decoder,
            n_skips=n_skips_decoder,
            activation_func=activation_func,
            n_pyramid=n_pyramid,
            deconv_type=decoder_type)

        # Move to device
        self.to(self.device)

    def forward(self, image, camera):
        '''
        Forwards an image through the network

        Args:
            image : tensor
                N x C x H x W tensor
            camera : float
                focal length baseline

        Returns:
            tensor : N x C x H x W left disparity
            tensor : N x C x H x W right disparity
            tensor : N x C x H x W left depth
            tensor : N x C x H x W right depth
        '''

        # Forward through network
        latent, skips = self.encoder(image)
        outputs = self.decoder(latent, skips)

        # Get disparity and depth output
        disparities = [self.scale_factor*output for output in outputs]

        self.disparities0 = []
        self.disparities1 = []
        self.depths0 = []
        self.depths1 = []
        # Output resolutions ordered in: 3, 2, 1, 0
        # Insert each output to the front to reverse 0, 1, 2, 3
        for disparity in disparities:
            disparity0, disparity1 = torch.chunk(disparity, chunks=2, dim=1)
            self.disparities0.insert(0, disparity0)
            self.disparities1.insert(0, disparity1)
            self.depths0.insert(0, camera/disparity0)
            self.depths1.insert(0, camera/disparity1)

        return self.disparities0[0], self.disparities1[0], self.depths0[0], self.depths1[0]

    def compute_loss(self,
                     train_image0,
                     train_image1,
                     w_color=0.15,
                     w_ssim=0.85,
                     w_smoothness=0.10,
                     w_left_right=1.00):
        '''
        Forwards an image through the network

        Args:
            train_image0 : tensor
                N x C x H x W left image
            train_image1 : tensor
                N x C x H x W right image
            w_color : float
                weight of color consistency term
            w_ssim : float
                weight of SSIM term
            w_smoothness : float
                weight of local smoothness
            w_left_right : float
                weight of left-right consistency

        Returns:
            float : loss
        '''

        # Create image pyramid
        images0 = [
            torch.nn.functional.interpolate(train_image0,
                scale_factor=1.0/2**s, mode='bilinear', align_corners=True)
            for s in range(self.n_pyramid)]
        images1 = [
            torch.nn.functional.interpolate(train_image1,
                scale_factor=1.0/2**s, mode='bilinear', align_corners=True)
            for s in range(self.n_pyramid)]
        # Create image reconstruction pyramid
        images0w = [
            net_utils.warp1d_horizontal(image1, -disparity0)
            for image1, disparity0 in zip(images1, self.disparities0)]
        images1w = [
            net_utils.warp1d_horizontal(image0, disparity1)
            for image0, disparity1 in zip(images0, self.disparities1)]
        # Create left-right disparity warping pyramid
        disparities0w = [
            net_utils.warp1d_horizontal(disparity1, -disparity0)
            for disparity0, disparity1 in zip(self.disparities0, self.disparities1)]
        disparities1w = [
            net_utils.warp1d_horizontal(disparity0, -disparity1)
            for disparity0, disparity1 in zip(self.disparities0, self.disparities1)]

        # For logging
        self.image0 = train_image0
        self.image1 = train_image1
        self.image0w = images0w[0]
        self.image1w = images1w[0]
        self.disparity0 = self.disparities0[0]
        self.disparity1 = self.disparities1[0]

        self.loss_color = 0.0
        self.loss_ssim = 0.0
        self.loss_smoothness = 0.0
        self.loss_left_right = 0.0
        for s in range(self.n_pyramid):
            # Compute image reconstruction loss function
            loss_color0 = loss_utils.color_consistency_loss_func(
                images0w[s], images0[s])
            loss_color1 = loss_utils.color_consistency_loss_func(
                images1w[s], images1[s])

            self.loss_color = self.loss_color+w_color*(loss_color0+loss_color1)

            loss_ssim0 = loss_utils.ssim_loss_func(images0w[s], images0[s])
            loss_ssim1 = loss_utils.ssim_loss_func(images1w[s], images1[s])

            self.loss_ssim = self.loss_ssim+w_ssim*(loss_ssim0+loss_ssim1)

            # Compute smoothness loss function
            loss_smoothness0 = loss_utils.smoothness_loss_func(
                self.disparities0[s], images0[s])
            loss_smoothness1 = loss_utils.smoothness_loss_func(
                self.disparities1[s], images1[s])

            self.loss_smoothness = \
                    self.loss_smoothness + \
                    w_smoothness * (loss_smoothness0+loss_smoothness1) / 2.0 ** s

            # Compute left right consistency loss function
            loss_left_right0 = loss_utils.left_right_consistency_loss_func(
                disparities0w[s], self.disparities0[s])
            loss_left_right1 = loss_utils.left_right_consistency_loss_func(
                disparities1w[s], self.disparities1[s])

            self.loss_left_right = \
                self.loss_left_right + \
                w_left_right*(loss_left_right0+loss_left_right1)

        self.loss = \
            self.loss_color + \
            self.loss_ssim + \
            self.loss_smoothness + \
            self.loss_left_right

        return self.loss

    def parameters(self):
        '''
        Moves model to device

        Args:
            device : torch.device
        '''

        return list(self.encoder.parameters()) + list(self.decoder.parameters())

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

    def to(self, device):
        '''
        Moves model to device

        Args:
            device : torch.device
        '''

        self.encoder.to(device)
        self.decoder.to(device)

    def restore_model(self, encoder_restore_path, decoder_restore_path):
        '''
        Restore weights of the model

        Args:
            encoder_restore_path : str
                path to encoder checkpoint
            decoder_restore_path : str
                path to decoder checkpoint

        Returns:
            None
        '''

        # Restore weights for encoder
        loaded_dict_encoder = torch.load(encoder_restore_path, map_location=self.device)
        self.encoder.load_state_dict(loaded_dict_encoder['model_state_dict'])

        # Restore weights for decoder
        loaded_dict_decoder = torch.load(decoder_restore_path, map_location=self.device)
        self.decoder.load_state_dict(loaded_dict_decoder['model_state_dict'])

        # Return the current step
        return loaded_dict_encoder['train_step']

    def log_summary(self, summary_writer, step):
        '''
        Logs to tensorboard

        Args:
            summary_writer : tensorboard.SummaryWriter
                path to encoder checkpoint
            step : str
                path to decoder checkpoint

        Returns:
            None
        '''

        with torch.no_grad():
            # Log events to summary
            summary_writer.add_scalar('loss', self.loss, global_step=step)
            summary_writer.add_scalar('loss_color', self.loss_color, global_step=step)
            summary_writer.add_scalar('loss_ssim', self.loss_ssim, global_step=step)
            summary_writer.add_scalar('loss_smoothness', self.loss_smoothness, global_step=step)
            summary_writer.add_scalar('loss_left_right', self.loss_left_right, global_step=step)

            # Log histogram
            summary_writer.add_histogram('disparity0', self.disparity0, global_step=step)
            summary_writer.add_histogram('disparity1', self.disparity1, global_step=step)
