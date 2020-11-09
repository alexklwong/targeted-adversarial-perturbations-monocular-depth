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
from perturb_main import run


parser = argparse.ArgumentParser()

# Training and validation input filepaths
parser.add_argument('--image_path',
    type=str, required=True, help='Path to list of image paths')
parser.add_argument('--preset_path',
    type=str, default='', help='Path to list of preset image paths')
parser.add_argument('--class_mask_path',
    type=str, default='', help='Path to list of class mask paths')
# Run settings
parser.add_argument('--n_height',
    type=int, default=settings.N_HEIGHT, help='Height of the image')
parser.add_argument('--n_width',
    type=int, default=settings.N_WIDTH, help='Width of the image')
parser.add_argument('--n_channel',
    type=int, default=settings.N_CHANNEL, help='Number of channels of the image')
parser.add_argument('--output_norm',
    type=float, default=settings.OUTPUT_NORM, help='Output norm of noise')
parser.add_argument('--n_step',
    type=int, default=settings.N_STEP, help='Number of steps to optimize')
parser.add_argument('--learning_rates',
    nargs='+', type=float, default=settings.LEARNING_RATES, help='Space delimited list of learning rates')
parser.add_argument('--learning_schedule',
    nargs='+', type=int, default=settings.LEARNING_SCHEDULE, help='Space delimited list to change learning rate')
# Target depth settings
parser.add_argument('--depth_method',
    type=str, default=settings.DEPTH_METHOD, help='Depth method available: %s' % settings.DEPTH_METHOD_AVAILABLE)
parser.add_argument('--depth_transform_func',
    nargs='+', type=str, default=settings.DEPTH_TRANSFORM_FUNC, help='Depth transform available: %s' % settings.DEPTH_TRANSFORM_FUNC_AVAILABLE)
parser.add_argument('--depth_transform_value',
    type=float, default=settings.DEPTH_TRANSFORM_VALUE, help='Value used in depth transform function')
parser.add_argument('--mask_constraint',
    type=str, default=settings.MASK_CONSTRAINT, help='Mask constraint available: %s' % settings.MASK_CONSTRAINT_AVAILABLE)
# Checkpoint settings
parser.add_argument('--checkpoint_path',
    type=str, default=settings.CHECKPOINT_PATH, help='Path to save checkpoints')
parser.add_argument('--depth_model_restore_path0',
    type=str, default=settings.DEPTH_MODEL_RESTORE_PATH0, help='Path to restore depth encoder or entire model checkpoint')
parser.add_argument('--depth_model_restore_path1',
    type=str, default=settings.DEPTH_MODEL_RESTORE_PATH1, help='Path to restore depth decoder checkpoint')
# Hardware settings
parser.add_argument('--device',
    type=str, default=settings.DEVICE, help='Device to use: gpu, cpu')

args = parser.parse_args()

if __name__ == '__main__':

    args.preset_path = None if args.preset_path == '' else args.preset_path
    args.class_mask_path = None if args.class_mask_path == '' else args.class_mask_path

    assert len(args.learning_rates) == len(args.learning_schedule) + 1

    assert args.output_norm > 0

    args.depth_method = args.depth_method.lower()
    assert args.depth_method in settings.DEPTH_METHOD_AVAILABLE

    args.depth_transform_func = [transform_func.lower() for transform_func in args.depth_transform_func]
    for transform_func in args.depth_transform_func:
        assert transform_func in settings.DEPTH_TRANSFORM_FUNC_AVAILABLE

    args.device = args.device.lower()
    if args.device not in [settings.GPU, settings.CPU, settings.CUDA]:
        args.device = settings.CUDA

    args.device = settings.CUDA if args.device == settings.GPU else args.device

    run(image_path=args.image_path,
        preset_path=args.preset_path,
        class_mask_path=args.class_mask_path,
        # Run settings
        n_height=args.n_height,
        n_width=args.n_width,
        n_channel=args.n_channel,
        output_norm=args.output_norm,
        n_step=args.n_step,
        learning_rates=args.learning_rates,
        learning_schedule=args.learning_schedule,
        # Target depth settings
        depth_method=args.depth_method,
        depth_transform_func=args.depth_transform_func,
        depth_transform_value=args.depth_transform_value,
        mask_constraint=args.mask_constraint,
        # Checkpoint settings
        checkpoint_path=args.checkpoint_path,
        depth_model_restore_path0=args.depth_model_restore_path0,
        depth_model_restore_path1=args.depth_model_restore_path1,
        # Hardware settings
        device=args.device)
