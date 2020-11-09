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
import os, time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import datasets, data_utils
import global_constants as settings
from log_utils import log
from monodepth_model import MonodepthModel


def train(train_image0_path,
          train_image1_path,
          train_camera_path,
          # Batch settings
          n_batch=settings.N_BATCH,
          n_height=settings.N_HEIGHT,
          n_width=settings.N_WIDTH,
          encoder_type=settings.ENCODER_TYPE,
          decoder_type=settings.DECODER_TYPE,
          activation_func=settings.ACTIVATION_FUNC,
          n_pyramid=settings.N_PYRAMID,
          # Training settings
          n_epoch=settings.N_EPOCH,
          learning_rates=settings.LEARNING_RATES,
          learning_schedule=settings.LEARNING_SCHEDULE,
          use_augment=settings.USE_AUGMENT,
          w_color=settings.W_COLOR,
          w_ssim=settings.W_SSIM,
          w_smoothness=settings.W_SMOOTHNESS,
          w_left_right=settings.W_LEFT_RIGHT,
          # Depth range settings
          scale_factor=settings.SCALE_FACTOR,
          # Checkpoint settings
          n_summary=settings.N_SUMMARY,
          n_checkpoint=settings.N_CHECKPOINT,
          checkpoint_path=settings.CHECKPOINT_PATH,
          # Hardware settings
          device=settings.DEVICE,
          n_thread=settings.N_THREAD):

    if device == settings.CUDA or device == settings.GPU:
        device = torch.device(settings.CUDA)
    else:
        device = torch.device(settings.CPU)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Set up checkpoint and event paths
    encoder_checkpoint_path = os.path.join(checkpoint_path, 'encoder-{}.pth')
    decoder_checkpoint_path = os.path.join(checkpoint_path, 'decoder-{}.pth')
    log_path = os.path.join(checkpoint_path, 'results.txt')
    event_path = os.path.join(checkpoint_path, 'events')

    # Read paths for training
    train_image0_paths = data_utils.read_paths(train_image0_path)
    train_image1_paths = data_utils.read_paths(train_image1_path)
    train_camera_paths = data_utils.read_paths(train_camera_path)

    assert len(train_image0_paths) == len(train_image1_paths)
    assert len(train_image0_paths) == len(train_camera_paths)

    n_train_sample = len(train_image0_paths)
    n_train_step = n_epoch*np.ceil(n_train_sample/n_batch).astype(np.int32)

    train_dataloader = torch.utils.data.DataLoader(
        datasets.ImagePairCameraDataset(
            train_image0_paths,
            train_image1_paths,
            train_camera_paths,
            shape=(n_height, n_width),
            augment=use_augment),
        batch_size=n_batch,
        shuffle=True,
        num_workers=n_thread,
        drop_last=False)

    # Build network
    model = MonodepthModel(
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        activation_func=activation_func,
        n_pyramid=n_pyramid,
        scale_factor=scale_factor,
        device=device)
    train_summary = SummaryWriter(event_path)
    parameters = model.parameters()
    n_param = sum(p.numel() for p in parameters)

    # Start training
    model.train()

    log('Network settings:', log_path)
    log('n_batch=%d  n_height=%d  n_width=%d  n_param=%d' %
        (n_batch, n_height, n_width, n_param), log_path)
    log('encoder_type=%s  decoder_type=%s  activation_func=%s  n_pyramid=%d' %
        (encoder_type, decoder_type, activation_func, n_pyramid), log_path)
    log('Training settings:', log_path)
    log('n_sample=%d  n_epoch=%d  n_step=%d' %
        (n_train_sample, n_epoch, n_train_step), log_path)
    log('learning_schedule=[%s]' %
        ', '.join('{}:{}'.format(l*(n_train_sample//n_batch), v, log_path)
        for l, v in zip([0] + learning_schedule, learning_rates)), log_path)
    log('use_augment=%s' % use_augment, log_path)
    log('w_color=%.2f  w_ssim=%.2f  w_smoothness=%.2f  w_left_right=%.2f' %
        (w_color, w_ssim, w_smoothness, w_left_right), log_path)
    log('Depth range settings:', log_path)
    log('scale_factor=%.2f' %
        (scale_factor), log_path)
    log('Checkpoint settings:', log_path)
    log('depth_model_checkpoint_path=%s' % checkpoint_path, log_path)

    learning_schedule.append(n_epoch)
    schedule_pos = 0
    train_step = 0
    time_start = time.time()
    log('Begin training...', log_path)
    for epoch in range(1, n_epoch+1):
        # Set learning rate schedule
        if epoch > learning_schedule[schedule_pos]:
            schedule_pos = schedule_pos + 1
        learning_rate = learning_rates[schedule_pos]
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)

        for train_image0, train_image1, train_camera in train_dataloader:
            train_step = train_step + 1
            # Fetch data
            if device.type == settings.CUDA:
                train_image0 = train_image0.cuda()
                train_image1 = train_image1.cuda()
                train_camera = train_camera.cuda()

            # Forward through the network
            model.forward(train_image0, train_camera)

            # Compute loss function
            loss = model.compute_loss(train_image0, train_image1,
                w_color=w_color,
                w_ssim=w_ssim,
                w_smoothness=w_smoothness,
                w_left_right=w_left_right)

            # Compute gradient and backpropagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (train_step % n_summary) == 0:
                model.log_summary(
                    summary_writer=train_summary,
                    step=train_step)

            # Log results and save checkpoints
            if (train_step % n_checkpoint) == 0:
                time_elapse = (time.time() - time_start) / 3600
                time_remain = (n_train_step - train_step) * time_elapse / train_step
                log('Step={:6}/{}  Loss={:.5f}  Time Elapsed={:.2f}h  Time Remaining={:.2f}h'.format(
                    train_step, n_train_step, loss.item(), time_elapse, time_remain), log_path)

                # Save checkpoints
                torch.save({
                    'train_step': train_step,
                    'model_state_dict': model.encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, encoder_checkpoint_path.format(train_step))
                torch.save({
                    'train_step': train_step,
                    'model_state_dict': model.decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, decoder_checkpoint_path.format(train_step))

    # Save checkpoints and close summary
    train_summary.close()
    torch.save({
        'train_step': train_step,
        'model_state_dict': model.encoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, encoder_checkpoint_path.format(train_step))
    torch.save({
        'train_step': train_step,
        'model_state_dict': model.decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, decoder_checkpoint_path.format(train_step))
