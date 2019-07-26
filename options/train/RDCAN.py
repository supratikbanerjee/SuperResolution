#### general settings
name: RACN_TEST_EXP_02_BSDS300_1conv_1RCAB
use_visdom: false
visdom_port: 8067
model: cnn
device: cuda

#### datasets
dataset:
  scale: 2
  train:
    name: DIV2K
    data_location: data/datasets/BSDS300/train/
    shuffle: true
    n_workers: 0  # per GPU
    batch_size: 25
    lr_size: 16
    repeat: 1
  test:
    name: Set5
    data_location: data/datasets/Set5/test/
    shuffle: false
    n_workers: 1  # per GPU
    batch_size: 1
    repeat: 1

#### network structures
network_G:
  model: RCAN
  n_resgroups: 2
  n_resblocks: 1
  num_features: 32
  in_channels: 3
  out_channels: 3
  res_scale: 1
  reduction: 16
  rgb_range: 255

#### training settings: learning rate scheme, loss
train:
  epoch: 100

  lr_G: !!float 1e-4
  weight_decay_G: 0
  beta1_G: 0.9
  beta2_G: 0.99

  lr_step: 10
  lr_gamma: 0.5

  pixel_criterion: l2
  pixel_weight: !!float 1e0

  manual_seed: 10
  val_freq: 5 # epoch

#### logger
logger:
  print_freq: 5 # epoch
  path: experiments/
  chkpt_freq: 2 # epoch
  