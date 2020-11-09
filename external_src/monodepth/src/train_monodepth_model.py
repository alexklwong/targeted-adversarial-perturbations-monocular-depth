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
import argparse
import global_constants as settings
from monodepth_main import train


parser = argparse.ArgumentParser()

# Training and validation input filepaths
parser.add_argument('--train_image0_path',
    type=str, required=True, help='Path to list of training image 0 paths')
parser.add_argument('--train_image1_path',
    type=str, required=True, help='Path to list of training image 1 paths')
parser.add_argument('--train_camera_path',
    type=str, required=True, help='Path to list of training camera parameter paths')
# Batch parameters
parser.add_argument('--n_batch',
    type=int, default=settings.N_BATCH, help='Number of samples per batch')
parser.add_argument('--n_height',
    type=int, default=settings.N_HEIGHT, help='Height of of sample')
parser.add_argument('--n_width',
    type=int, default=settings.N_WIDTH, help='Width of each sample')
parser.add_argument('--encoder_type',
    type=str, default=settings.ENCODER_TYPE, help='Encoder type: %s' % settings.ENCODER_TYPE_AVAILABLE)
parser.add_argument('--decoder_type',
    type=str, default=settings.DECODER_TYPE, help='Decoder type: %s' % settings.DECODER_TYPE_AVAILABLE)
parser.add_argument('--activation_func',
    type=str, default=settings.ACTIVATION_FUNC, help='Activation func: %s' % settings.ACTIVATION_FUNC_AVAILABLE)
parser.add_argument('--n_pyramid',
    type=int, default=settings.N_PYRAMID, help='Number of levels in image pyramid')
# Training settings
parser.add_argument('--n_epoch',
    type=int, default=settings.N_EPOCH, help='Number of epochs for training')
parser.add_argument('--learning_rates',
    nargs='+', type=float, default=settings.LEARNING_RATES, help='Space delimited list of learning rates')
parser.add_argument('--learning_schedule',
    nargs='+', type=int, default=settings.LEARNING_SCHEDULE, help='Space delimited list to change learning rate')
parser.add_argument('--use_augment',
    action='store_true', help='If set, then use data augmentation')
parser.add_argument('--w_color',
    type=float, default=settings.W_COLOR, help='Weight of image reconstruction loss')
parser.add_argument('--w_ssim',
    type=float, default=settings.W_SSIM, help='Weight of ssim loss')
parser.add_argument('--w_smoothness',
    type=float, default=settings.W_SMOOTHNESS, help='Weight of smoothness loss')
parser.add_argument('--w_left_right',
    type=float, default=settings.W_LEFT_RIGHT, help='Weight of left-right consistency loss')
# Depth range settings
parser.add_argument('--scale_factor',
    type=float, default=settings.SCALE_FACTOR, help='Scale factor for disparity')
# Checkpoint settings
parser.add_argument('--n_summary',
    type=int, default=settings.N_SUMMARY, help='Number of iterations for logging summary')
parser.add_argument('--n_checkpoint',
    type=int, default=settings.N_CHECKPOINT, help='Number of iterations for each checkpoint')
parser.add_argument('--checkpoint_path',
    type=str, default=settings.CHECKPOINT_PATH, help='Path to save checkpoints')
# Hardware settings
parser.add_argument('--device',
    type=str, default=settings.DEVICE, help='Device to use: gpu, cpu')
parser.add_argument('--n_thread',
    type=int, default=settings.N_THREAD, help='Number of threads for fetching')


args = parser.parse_args()

if __name__ == '__main__':

    assert len(args.learning_rates) == len(args.learning_schedule) + 1

    args.encoder_type = args.encoder_type.lower()
    assert args.encoder_type in settings.ENCODER_TYPE_AVAILABLE

    args.decoder_type = args.decoder_type.lower()
    assert args.decoder_type in settings.DECODER_TYPE_AVAILABLE

    args.activation_func = args.activation_func.lower()
    assert args.activation_func in settings.ACTIVATION_FUNC_AVAILABLE

    args.device = args.device.lower()
    if args.device not in [settings.GPU, settings.CPU, settings.CUDA]:
        args.device = settings.CUDA

    args.device = settings.CUDA if args.device == settings.GPU else args.device

    train(train_image0_path=args.train_image0_path,
          train_image1_path=args.train_image1_path,
          train_camera_path=args.train_camera_path,
          # Batch settings
          n_batch=args.n_batch,
          n_height=args.n_height,
          n_width=args.n_width,
          encoder_type=args.encoder_type,
          decoder_type=args.decoder_type,
          activation_func=args.activation_func,
          n_pyramid=args.n_pyramid,
          # Training settings
          n_epoch=args.n_epoch,
          learning_rates=args.learning_rates,
          learning_schedule=args.learning_schedule,
          use_augment=args.use_augment,
          w_color=args.w_color,
          w_ssim=args.w_ssim,
          w_smoothness=args.w_smoothness,
          w_left_right=args.w_left_right,
          # Depth range settings
          scale_factor=args.scale_factor,
          # Checkpoint settings
          n_summary=args.n_summary,
          n_checkpoint=args.n_checkpoint,
          checkpoint_path=args.checkpoint_path,
          # Hardware settings
          device=args.device,
          n_thread=args.n_thread)
