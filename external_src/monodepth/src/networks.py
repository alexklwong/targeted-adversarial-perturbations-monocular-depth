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


'''
Network architecture blocks
'''
class Conv2d(torch.nn.Module):
    '''
    2D convolution class

    Args:
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        kernel_size : int
            size of kernel
        activation_func : func
            activation function after convolution
        stride : int
            stride of convolution
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 padding=1,
                 stride=1):
        super(Conv2d, self).__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False),
            activation_func)

    def forward(self, x):
        return self.conv(x)


class ResNetBlock(torch.nn.Module):
    '''
    Basic ResNet block class

    Args:
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        activation_func : func
            activation function after convolution
        stride : int
            stride of convolution
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 stride=1):
        super(ResNetBlock, self).__init__()

        self.activation_func = activation_func

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            self.activation_func)

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            self.activation_func)

        self.projection = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        # Perform 2 convolutions
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        # Perform projection if (1) shape does not match (2) channels do not match
        in_shape = list(x.shape)
        out_shape = list(conv2.shape)
        if in_shape[2:4] != out_shape[2:4] or in_shape[1] != out_shape[1]:
            X = self.projection(x)
        else:
            X = x
        # f(x) + x
        return self.activation_func(conv2+X)


class ResNetBottleneckBlock(torch.nn.Module):
    '''
    ResNet bottleneck block class

    Args:
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        activation_func : func
            activation function after convolution
        stride : int
            stride of convolution
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 stride=1):
        super(ResNetBottleneckBlock, self).__init__()

        self.activation_func = activation_func

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            self.activation_func)

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            self.activation_func)

        self.conv3 = torch.nn.Conv2d(
            out_channels, 4 * out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.projection = torch.nn.Conv2d(
            in_channels, 4 * out_channels, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        # Perform 3 convolutions
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        # Perform projection if (1) shape does not match (2) channels do not match
        in_shape = list(x.shape)
        out_shape = list(conv2.shape)
        if in_shape[2:4] != out_shape[2:4] or in_shape[1] != out_shape[1]:
            X = self.projection(x)
        else:
            X = x
        # f(x) + x
        return self.activation_func(conv3+X)


class VGGNetBlock(torch.nn.Module):
    '''
    VGGNet block class

    Args:
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        activation_func : func
            activation function after convolution
        n_conv : int
            number of convolution layers
        stride : int
            stride of convolution
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 n_conv=1,
                 stride=1):
        super(VGGNetBlock, self).__init__()

        layers = []
        for n in range(n_conv):
            layers.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    activation_func))
            in_channels = out_channels

        layers.append(
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                activation_func))

        self.conv_block = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)


class TransposeConv2d(torch.nn.Module):
    '''
    Transpose convolution class

    Args:
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        activation_func : func
            activation function after convolution
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True)):
        super(TransposeConv2d, self).__init__()

        self.deconv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, out_channels,
                kernel_size=3, stride=2, padding=1, output_padding=1),
            activation_func)

    def forward(self, x):
        return self.deconv(x)


class UpConv2d(torch.nn.Module):
    '''
    Up-convolution (upsample + convolution) block class

    Args:
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        activation_func : func
            activation function after convolution
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True)):
        super(UpConv2d, self).__init__()

        self.deconv = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            activation_func)

    def forward(self, x):
        return self.deconv(x)


class DecoderBlock(torch.nn.Module):
    '''
    Decoder block with skip connections (U-Net)

    Args:
        in_channels : int
            number of input channels
        skip_channels : int
            number of skip connection channels
        out_channels : int
            number of output channels
        activation_func : func
            activation function after convolution
        deconv_type : str
            deconvolution types: transpose, up
    '''
    def __init__(self,
                 in_channels,
                 skip_channels,
                 out_channels,
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 deconv_type='transpose'):
        super(DecoderBlock, self).__init__()

        self.skip_channels = skip_channels

        if deconv_type == 'transpose':
            self.deconv = TransposeConv2d(in_channels, out_channels, activation_func)
        elif deconv_type == 'up':
            self.deconv = UpConv2d(in_channels, out_channels, activation_func)

        concat_channels = skip_channels+out_channels
        self.conv = torch.nn.Conv2d(concat_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, skip=None):
        deconv = self.deconv(x)
        if self.skip_channels > 0:
            concat = torch.cat([deconv, skip], dim=1)
        else:
            concat = deconv
        return self.conv(concat)


'''
Encoder architectures
'''
class ResNetEncoder(torch.nn.Module):
    '''
    ResNet encoder with skip connections

    Args:
        n_layer : int
            architecture type based on layers: 18, 34
        input_channels : int
            number of input channels
        n_filters : int
            number of filters per block
        activation_func : func
            activation function after convolution
    '''
    def __init__(self,
                 n_layer,
                 input_channels=3,
                 n_filters=[32, 64, 128, 256, 256],
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True)):
        super(ResNetEncoder, self).__init__()

        use_bottleneck = False
        if n_layer == 18:
            n_blocks = [2, 2, 2, 2]
            resnet_block = ResNetBlock
        elif n_layer == 34:
            n_blocks = [3, 4, 6, 3]
            resnet_block = ResNetBlock
        elif n_layer == 50:
            n_blocks = [3, 4, 6, 3]
            use_bottleneck = True
            resnet_block = ResNetBottleneckBlock
        else:
            raise ValueError('Only supports 18, 34, 50 layer architecture')

        assert(len(n_filters) == len(n_blocks)+1)

        in_channels, out_channels = [input_channels, n_filters[0]]
        self.conv0 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False),
            activation_func)

        self.max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        blocks1 = []
        in_channels, out_channels = [n_filters[0], n_filters[1]]
        for n in range(n_blocks[0]):
            if n == 0:
                blocks1.append(resnet_block(in_channels, out_channels, activation_func, stride=1))
            else:
                in_channels = 4*out_channels if use_bottleneck else out_channels
                blocks1.append(resnet_block(in_channels, out_channels, activation_func, stride=1))
        self.blocks1 = torch.nn.Sequential(*blocks1)

        blocks2 = []
        in_channels, out_channels = [n_filters[1], n_filters[2]]
        for n in range(n_blocks[1]):
            if n == 0:
                in_channels = 4*in_channels if use_bottleneck else in_channels
                blocks2.append(resnet_block(in_channels, out_channels, activation_func, stride=2))
            else:
                in_channels = 4*out_channels if use_bottleneck else out_channels
                blocks2.append(resnet_block(in_channels, out_channels, activation_func, stride=1))
        self.blocks2 = torch.nn.Sequential(*blocks2)

        blocks3 = []
        in_channels, out_channels = [n_filters[2], n_filters[3]]
        for n in range(n_blocks[2]):
            if n == 0:
                in_channels = 4*in_channels if use_bottleneck else in_channels
                blocks3.append(resnet_block(in_channels, out_channels, activation_func, stride=2))
            else:
                in_channels = 4*out_channels if use_bottleneck else out_channels
                blocks3.append(resnet_block(in_channels, out_channels, activation_func, stride=1))
        self.blocks3 = torch.nn.Sequential(*blocks3)

        blocks4 = []
        in_channels, out_channels = [n_filters[3], n_filters[4]]
        for n in range(n_blocks[3]):
            if n == 0:
                in_channels = 4*in_channels if use_bottleneck else in_channels
                blocks4.append(resnet_block(in_channels, out_channels, activation_func, stride=2))
            else:
                in_channels = 4*out_channels if use_bottleneck else out_channels
                blocks4.append(resnet_block(in_channels, out_channels, activation_func, stride=1))
        self.blocks4 = torch.nn.Sequential(*blocks4)

    def forward(self, x):
        layers = [x]
        # Resolution 1/1 -> 1/2
        layers.append(self.conv0(layers[-1]))
        # Resolution 1/2 -> 1/4
        max_pool = self.max_pool(layers[-1])
        layers.append(self.blocks1(max_pool))
        # Resolution 1/4 -> 1/8
        layers.append(self.blocks2(layers[-1]))
        # Resolution 1/8 -> 1/16
        layers.append(self.blocks3(layers[-1]))
        # Resolution 1/16 -> 1/32
        layers.append(self.blocks4(layers[-1]))

        return layers[-1], layers[0:-1]


class VGGNetEncoder(torch.nn.Module):
    '''
    VGGNet encoder with skip connections

    Args:
        n_layer : int
            architecture type based on layers: 8, 11, 13
        input_channels : int
            number of input channels
        n_filters : int
            number of filters per block
        activation_func : func
            activation function after convolution
    '''
    def __init__(self,
                 n_layer,
                 input_channels=3,
                 n_filters=[32, 64, 128, 256, 256],
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True)):
        super(VGGNetEncoder, self).__init__()

        if n_layer == 8:
            n_convs = [1, 1, 1, 1, 1]
        elif n_layer == 11:
            n_convs = [1, 1, 2, 2, 2]
        elif n_layer == 13:
            n_convs = [2, 2, 2, 2, 2]
        else:
            raise ValueError('Only supports 8, 11, 13 layer architecture')

        assert(len(n_filters) == len(n_convs))

        # Resolution 1/1 -> 1/2: 32 filters
        in_channels, out_channels = [input_channels, n_filters[0]]
        stride = 1 if n_convs[0]-1 > 0 else 2
        conv0 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=False),
            activation_func)

        if n_convs[0]-1 > 0:
            self.conv1 = torch.nn.Sequential(
                conv0,
                VGGNetBlock(out_channels, out_channels, activation_func, n_conv=n_convs[0]-1, stride=2))
        else:
            self.conv1 = conv0

        # Resolution 1/2 -> 1/4: 64 filters
        in_channels, out_channels = [n_filters[0], n_filters[1]]
        self.conv2 = VGGNetBlock(in_channels, out_channels, activation_func, n_conv=n_convs[1], stride=2)

        # Resolution 1/4 -> 1/8: 128 filters
        in_channels, out_channels = [n_filters[1], n_filters[2]]
        self.conv3 = VGGNetBlock(in_channels, out_channels, activation_func, n_conv=n_convs[2], stride=2)

        # Resolution 1/8 -> 1/16: 256 filters
        in_channels, out_channels = [n_filters[2], n_filters[3]]
        self.conv4 = VGGNetBlock(in_channels, out_channels, activation_func, n_conv=n_convs[3], stride=2)

        # Resolution 1/16 -> 1/32: 256 filters
        in_channels, out_channels = [n_filters[3], n_filters[4]]
        self.conv5 = VGGNetBlock(in_channels, out_channels, activation_func, n_conv=n_convs[4], stride=2)

    def forward(self, x):
        layers = [x]
        # Resolution 1/1 -> 1/2
        layers.append(self.conv1(layers[-1]))
        # Resolution 1/2 -> 1/4
        layers.append(self.conv2(layers[-1]))
        # Resolution 1/4 -> 1/8
        layers.append(self.conv3(layers[-1]))
        # Resolution 1/8 -> 1/32
        layers.append(self.conv4(layers[-1]))
        # Resolution 1/16 -> 1/32
        layers.append(self.conv5(layers[-1]))

        return layers[-1], layers[0:-1]


'''
Decoder architectures
'''
class DisparityDecoder(torch.nn.Module):
    '''
    Decoder to predict disparity

    Args:
        input_channels : int
            number of input channels
        output_channels : int
            number of output channels
        n_pyramid : int
            depth of pyramid
        n_filters : int
            number of filters per block
        n_skips : int
            number of channels in skip connections
        activation_func : func
            activation function after convolution
        deconv_type : str
            transpose or up convolution
    '''
    def __init__(self,
                 input_channels=256,
                 output_channels=2,
                 n_pyramid=4,
                 n_filters=[256, 128, 64, 32, 16],
                 n_skips=[256, 128, 64, 32, 0],
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 deconv_type='transpose'):
        super(DisparityDecoder, self).__init__()

        network_depth = 5
        assert(n_pyramid > 0 and n_pyramid < network_depth)
        assert(len(n_filters) == network_depth)
        assert(len(n_skips) == network_depth)

        self.n_pyramid = n_pyramid

        # Resolution 1/32 -> 1/16
        in_channels, skip_channels, out_channels = [input_channels, n_skips[0], n_filters[0]]
        self.deconv4 = DecoderBlock(in_channels, skip_channels, out_channels,
            activation_func=activation_func, deconv_type=deconv_type)

        # Resolution 1/16 -> 1/8
        in_channels, skip_channels, out_channels = [n_filters[0], n_skips[1], n_filters[1]]
        self.deconv3 = DecoderBlock(in_channels, skip_channels, out_channels,
            activation_func=activation_func, deconv_type=deconv_type)
        self.output3 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channels, output_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.Sigmoid())

        # Resolution 1/8 -> 1/4
        in_channels, skip_channels, out_channels = [n_filters[1], n_skips[2], n_filters[2]]
        if self.n_pyramid > 3:
            skip_channels = skip_channels+output_channels
        self.deconv2 = DecoderBlock(in_channels, skip_channels, out_channels,
            activation_func=activation_func, deconv_type=deconv_type)
        self.output2 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channels, output_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.Sigmoid())

        # Resolution 1/4 -> 1/2
        in_channels, skip_channels, out_channels = [n_filters[2], n_skips[3], n_filters[3]]
        if self.n_pyramid > 2:
            skip_channels = skip_channels+output_channels
        self.deconv1 = DecoderBlock(in_channels, skip_channels, out_channels,
            activation_func=activation_func, deconv_type=deconv_type)
        self.output1 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channels, output_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.Sigmoid())

        # Resolution 1/2 -> 1/1
        in_channels, skip_channels, out_channels = [n_filters[3], n_skips[4], n_filters[4]]

        if self.n_pyramid > 1:
            skip_channels = skip_channels+output_channels
        self.deconv0 = DecoderBlock(in_channels, skip_channels, out_channels,
            activation_func=activation_func, deconv_type=deconv_type)
        self.output0 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channels, output_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.Sigmoid())

    def forward(self, x, skips):
        layers = [x]
        outputs = []

        # Resolution 1/32 -> 1/16
        n = len(skips)-1
        layers.append(self.deconv4(layers[-1], skips[n]))

        # Resolution 1/16 -> 1/8
        n = n-1
        layers.append(self.deconv3(layers[-1], skips[n]))
        if self.n_pyramid > 3:
            outputs.append(self.output3(layers[-1]))
            upsample_output3 = torch.nn.functional.interpolate(outputs[-1],
                scale_factor=2, mode='bilinear', align_corners=True)

        # Resolution 1/8 -> 1/4
        n = n-1
        skip = torch.cat([skips[n], upsample_output3], dim=1) if self.n_pyramid > 3 else skips[n]
        layers.append(self.deconv2(layers[-1], skip))
        if self.n_pyramid > 2:
            outputs.append(self.output2(layers[-1]))
            upsample_output2 = torch.nn.functional.interpolate(outputs[-1],
                scale_factor=2, mode='bilinear', align_corners=True)

        # Resolution 1/4 -> 1/2
        n = n-1
        skip = torch.cat([skips[n], upsample_output2], dim=1) if self.n_pyramid > 2 else skips[n]
        layers.append(self.deconv1(layers[-1], skip))
        if self.n_pyramid > 1:
            outputs.append(self.output1(layers[-1]))
            upsample_output1 = torch.nn.functional.interpolate(outputs[-1],
                scale_factor=2, mode='bilinear', align_corners=True)

        # Resolution 1/2 -> 1/1
        if self.n_pyramid > 1:
            layers.append(self.deconv0(layers[-1], upsample_output1))
        else:
            layers.append(self.deconv0(layers[-1]))

        outputs.append(self.output0(layers[-1]))
        return outputs
