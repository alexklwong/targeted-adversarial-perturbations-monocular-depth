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
import os, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
from PIL import Image
import torch
import datasets, data_utils, eval_utils, net_utils
import global_constants as settings
from log_utils import log
from depth_model import DepthModel
from perturb_model import PerturbationsModel


def run(image_path,
        preset_path=None,
        class_mask_path=None,
        # Run settings
        n_height=settings.N_HEIGHT,
        n_width=settings.N_WIDTH,
        n_channel=settings.N_CHANNEL,
        output_norm=settings.OUTPUT_NORM,
        n_step=settings.N_STEP,
        learning_rates=settings.LEARNING_RATES,
        learning_schedule=settings.LEARNING_SCHEDULE,
        # Target depth settings
        depth_method=settings.DEPTH_METHOD,
        depth_transform_func=settings.DEPTH_TRANSFORM_FUNC,
        depth_transform_value=settings.DEPTH_TRANSFORM_VALUE,
        mask_constraint=settings.MASK_CONSTRAINT,
        # Checkpoint settings
        checkpoint_path=settings.CHECKPOINT_PATH,
        depth_model_restore_path0=settings.DEPTH_MODEL_RESTORE_PATH0,
        depth_model_restore_path1=settings.DEPTH_MODEL_RESTORE_PATH1,
        # Hardware settings
        device=settings.DEVICE):

    if device == settings.CUDA or device == settings.GPU:
        device = torch.device(settings.CUDA)
    else:
        device = torch.device(settings.CPU)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Set up logging and output paths
    log_path = os.path.join(checkpoint_path, 'results.txt')
    output_path = os.path.join(checkpoint_path, 'outputs')

    # Read paths
    image_paths = data_utils.read_paths(image_path)
    n_sample = len(image_paths)

    if preset_path is not None:
        preset_paths = data_utils.read_paths(preset_path)[0:n_sample]
    else:
        preset_paths = [None] * n_sample

    if class_mask_path is not None:
        class_mask_paths = data_utils.read_paths(class_mask_path)
        assert(n_sample == len(class_mask_paths))
    else:
        class_mask_paths = [None] * n_sample

    # Set up depth model
    depth_model = DepthModel(method=depth_method, device=device)

    # Restore depth prediction model
    depth_model.restore_model(depth_model_restore_path0, depth_model_restore_path1)

    dataloader = torch.utils.data.DataLoader(
        datasets.TargetedAdversarialMonocularDataset(
            image_paths,
            preset_paths,
            class_mask_paths,
            shape=(n_height, n_width)),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False)

    log('Run settings:', log_path)
    log('n_height=%d  n_width=%d  n_channel=%d  output_norm=%.3f' %
        (n_height, n_width, n_channel, output_norm), log_path)
    log('learning_schedule=[%s]  n_step=%d' %
        (', '.join('{}:{:.1E}'.format(l, v)
        for l, v in zip([0] + learning_schedule, learning_rates)), n_step), log_path)
    log('depth_method=%s' %
        (depth_method), log_path)
    log('depth_transform_func=%s  depth_transform_value=%.2f' %
        (depth_transform_func, depth_transform_value), log_path)
    log('preset_path=%s' %
        (preset_path if preset_path is not None else False), log_path)
    log('class_mask_path=%s' %
        (class_mask_path if class_mask_path is not None else False), log_path)

    log('Checkpoint settings:', log_path)
    log('depth_model_restore_path0=%s' % depth_model_restore_path0, log_path)
    log('depth_model_restore_path1=%s' % depth_model_restore_path1, log_path)

    log('Evaluating...', log_path)

    # Initialize lists to store outputs
    images_origin = []
    depths_origin = []
    depths_target = []
    class_masks = []
    noises_output = []
    images_output = []
    depths_output = []

    time_start = time.time()
    learning_schedule.append(n_step)

    for idx, (image, preset, camera, class_mask) in enumerate(dataloader):

        if device.type == settings.CUDA:
            image = image.cuda()
            preset = preset.cuda()
            class_mask = class_mask.cuda()
            camera = camera.cuda()

        # Initialize perturbations
        model = PerturbationsModel(
            n_height=n_height,
            n_width=n_width,
            n_channel=3,
            output_norm=output_norm,
            device=device)

        # Forward image through depth network
        depth_origin = depth_model.forward(image, camera)

        if preset_path is None:
            depth_target = depth_origin
        else:
            depth_target = depth_model.forward(preset, camera)

        depth_target = net_utils.depth_transform(
            depth_target,
            class_mask,
            transform_type=depth_transform_func,
            value=depth_transform_value)

        # Store original image, depth, and class mask
        images_origin.append(
            np.transpose(np.squeeze(image.detach().cpu().numpy()), (1, 2, 0)))
        depths_origin.append(
            np.squeeze(depth_origin.detach().cpu().numpy()))
        depths_target.append(
            np.squeeze(depth_target.detach().cpu().numpy()))
        class_masks.append(
            np.squeeze(class_mask.detach().cpu().numpy()))

        schedule_pos = 0
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rates[0])
        for step in range(1, n_step + 1):

            if step > learning_schedule[schedule_pos]:
                schedule_pos = schedule_pos + 1
                learning_rate = learning_rates[schedule_pos]

                # Update optimizer learning rates
                for g in optimizer.param_groups:
                    g['lr'] = learning_rate

            # Compute and apply the perturbations to the image
            noise_output, image_output = model.forward(image)

            if mask_constraint == 'within_mask':
                noise_output *= class_mask
            elif mask_constraint == 'out_of_mask':
                noise_output *= (1.0 - class_mask)

            if mask_constraint != 'none':
                image_output = torch.clamp(image + noise_output, 0.0, 1.0)

            depth_output = depth_model.forward(image_output, camera)

            loss = model.compute_loss(
                depth_output=depth_output,
                depth_target=depth_target)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            sys.stdout.write(
                'Sample={:3}/{:3}  Step={:3}/{:3}  Loss={:.5f}\r'.format(
                    idx + 1, n_sample, step, n_step, loss.item()))
            sys.stdout.flush()

        # Save image, noise and depth outputs
        noises_output.append(
            np.transpose(np.squeeze(noise_output.detach().cpu().numpy()), (1, 2, 0)))
        images_output.append(
            np.transpose(np.squeeze(image_output.detach().cpu().numpy()), (1, 2, 0)))
        depths_output.append(
            np.squeeze(depth_output.detach().cpu().numpy()))

        # Log results
        time_elapse = (time.time() - time_start) / 3600
        log('Sample={:3}/{:3}  Loss={:.5f}  Time Elapsed={:.2f}h'.format(
            idx + 1, n_sample, loss.item(), time_elapse), log_path)

    # Evaluate results
    evaluate(noises_output, depths_output, depths_target, class_masks, log_path)

    log('Storing image and depth outputs into {}'.format(output_path))
    image_origin_path = os.path.join(output_path, 'image_origin')
    depth_origin_path = os.path.join(output_path, 'depth_origin')
    depth_target_path = os.path.join(output_path, 'depth_target')
    image_output_path = os.path.join(output_path, 'image_output')
    noise_output_path = os.path.join(output_path, 'noise_output')
    depth_output_path = os.path.join(output_path, 'depth_output')

    # Create output paths if they don't exist
    output_paths = [
        output_path,
        image_origin_path,
        depth_origin_path,
        depth_target_path,
        image_output_path,
        noise_output_path,
        depth_output_path
    ]

    for path in output_paths:
        if not os.path.exists(path):
            os.makedirs(path)

    outputs = zip(
        images_origin,
        depths_origin,
        depths_target,
        images_output,
        noises_output,
        depths_output)

    for idx, output in enumerate(outputs):
        image_origin, depth_origin, depth_target, image_output, noise_output, depth_output = output
        image_filename = '{:05d}.png'.format(idx)
        numpy_filename = '{:05d}.npy'.format(idx)

        # Save to disk
        Image.fromarray(np.uint8(255.0 * image_origin)).save(os.path.join(image_origin_path, image_filename))
        np.save(os.path.join(depth_origin_path, numpy_filename), depth_origin)
        np.save(os.path.join(depth_target_path, numpy_filename), depth_target)
        Image.fromarray(np.uint8(255.0 * image_output)).save(os.path.join(image_output_path, image_filename))
        np.save(os.path.join(noise_output_path, numpy_filename), noise_output)
        np.save(os.path.join(depth_output_path, numpy_filename), depth_output)

def evaluate(noises_output,
             depths_output,
             depths_target,
             class_masks,
             log_path):

    n_sample = len(noises_output)

    # Noise metrics
    noise_l0pix = np.zeros(n_sample)
    noise_l1pix = np.zeros(n_sample)

    # Target depth metrics
    depth_absrel = np.zeros(n_sample)
    depth_absrel_class = np.zeros(n_sample)
    depth_absrel_nonclass = np.zeros(n_sample)

    data = zip(noises_output, depths_output, depths_target, class_masks)

    for idx, (noise_output, depth_output, depth_target, class_mask) in enumerate(data):

        # Compute noise, depth metrics
        noise_l0pix[idx] = eval_utils.lp_norm(noise_output, p=0)
        noise_l1pix[idx] = eval_utils.lp_norm(noise_output, p=1, axis=-1)

        # Depth error metrics over entire image space
        depth_absrel[idx] = eval_utils.abs_rel_err(depth_output, depth_target)

        # Depth error metrics for regions belonging to specific class
        mask = class_mask > 0

        if np.sum(mask) == 0:
            depth_absrel_class[idx] = -1.0
        else:
            depth_absrel_class[idx] = eval_utils.abs_rel_err(depth_output[mask], depth_target[mask])

        # Depth error metrics for regions not belonging to specific class
        mask = class_mask == 0

        if np.sum(mask) == 0:
            depth_absrel_nonclass[idx] = -1.0
        else:
            depth_absrel_nonclass[idx] = eval_utils.abs_rel_err(depth_output[mask], depth_target[mask])

    # Image, noise metrics
    noise_l0pix = np.mean(noise_l0pix)
    noise_l1pix = np.mean(noise_l1pix)

    # Depth error metrics over entire image space
    depth_absrel_std = np.std(depth_absrel)
    depth_absrel = np.mean(depth_absrel)

    # Depth error metrics for regions belonging to specific class
    depth_absrel_class = depth_absrel_class[depth_absrel_class != -1.0]
    depth_absrel_std_class = np.std(depth_absrel_class)
    depth_absrel_class = np.mean(depth_absrel_class)

    # Depth error metrics for regions not belonging to specific class
    depth_absrel_nonclass = depth_absrel_nonclass[depth_absrel_nonclass != -1.0]
    depth_absrel_std_nonclass = np.std(depth_absrel_nonclass)
    depth_absrel_nonclass = np.mean(depth_absrel_nonclass)

    log('Validation results:', log_path)
    log('{:<14}  {:>8}  {:>8}'.format('Noise:', 'L0 Pixel', 'L1 Pixel'), log_path)
    log('{:<14}  {:8.4f}  {:8.4f}'.format(
        '', noise_l0pix, noise_l1pix), log_path)
    log('{:<14}  {:>8}  {:>8}'.format(
        'Depth (All):', 'AbsRel', '+/-'), log_path)
    log('{:<14}  {:8.4f}  {:>8.4f}'.format(
        '', depth_absrel, depth_absrel_std), log_path)
    log('{:<14}  {:>8}  {:>8}'.format(
        'Depth (Class):', 'AbsRel', '+/-'), log_path)
    log('{:<14}  {:8.4f}  {:>8.4f}'.format(
        '', depth_absrel_class, depth_absrel_std_class), log_path)
    log('{:<14}  {:>8}  {:>8}'.format(
        'Depth (Rest):', 'AbsRel', '+/-'), log_path)
    log('{:<14}  {:8.4f}  {:>8.4f}'.format(
        '', depth_absrel_nonclass, depth_absrel_std_nonclass), log_path)
