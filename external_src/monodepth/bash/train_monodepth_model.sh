export CUDA_VISIBLE_DEVICES=0

python src/train_monodepth_model.py \
--train_image0_path training/kitti_eigen_train_image0.txt \
--train_image1_path training/kitti_eigen_train_image1.txt \
--train_camera_path training/kitti_eigen_train_camera.txt \
--n_batch 8 \
--n_height 192 \
--n_width 640 \
--encoder_type resnet50 \
--decoder_type up \
--n_epoch 30 \
--learning_rates 1.0e-4 0.5e-4 0.25e-4 \
--learning_schedule 18 24 \
--use_augment \
--w_color 0.15 \
--w_ssim 0.85 \
--w_smoothness 0.10 \
--w_left_right 1.00 \
--scale_factor 0.30 \
--n_summary 1000 \
--n_checkpoint 5000 \
--checkpoint_path monodepth_models/resnet50 \
--device gpu \
--n_thread 8
