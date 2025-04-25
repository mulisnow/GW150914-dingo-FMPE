import yaml

train_settings = """
data:
  waveform_dataset_path: 01_training_data/waveform_dataset/waveform_dataset.hdf5  # Contains intrinsic waveforms
  train_fraction: 0.95
  window:  # Needed to calculate window factor for simulated data
    type: tukey
    f_s: 4096
    T: 2.0
    roll_off: 0.4
  detectors:
    - H1
    - L1
  extrinsic_prior:  # Sampled at train time
    dec: bilby.core.prior.analytical.DeltaFunction(-1.2616009712219238)
    ra: bilby.core.prior.analytical.DeltaFunction(1.4557750225067139)
    geocent_time: bilby.core.prior.analytical.DeltaFunction(0.011423417367041111)
    psi: bilby.core.prior.analytical.DeltaFunction(1.2124483585357666)
    luminosity_distance: bilby.core.prior.analytical.DeltaFunction(488.2327880859375)
  ref_time: 1126259462.391
  inference_parameters:
  - chirp_mass
  - mass_ratio

# Model architecture
model:
  # kwargs for neural spline flow
  posterior_model_type: normalizing_flow
  posterior_kwargs:
    num_flow_steps: 5 # 30
    base_transform_kwargs:
      hidden_dim: 64 # 1024
      num_transform_blocks: 5
      activation: elu
      dropout_probability: 0.0
      batch_norm: True
      num_bins: 8
      base_transform_type: rq-coupling
  # kwargs for embedding net
  embedding_kwargs:
    output_dim: 64 # 128
    hidden_dims: [1024, 512, 256, 64]
    activation: elu
    dropout: 0.0
    batch_norm: True
    svd:
      num_training_samples: 1000
      num_validation_samples: 100
      size: 50

# The first stage (and only) stage of training.
training:
  stage_0:
    epochs: 15
    asd_dataset_path: 01_training_data/asd_dataset/asd_GW150914.hdf5
    freeze_rb_layer: True
    optimizer:
      type: adam
      lr: 0.0001
    scheduler:
      type: cosine
      T_max: 15
    batch_size: 64
  # stage_1:
  #   epochs: 5
  #   asd_dataset_path: 01_training_data/asd_dataset/asd_GW150914.hdf5
  #   freeze_rb_layer: False
  #   optimizer:
  #     type: adam
  #     lr: 1.e-5
  #   scheduler:
  #     type: cosine
  #     T_max: 5
  #   batch_size: 64

# Local settings for training that have no impact on the final trained network.
local:
  device: cpu  # [cpu, cuda] Set this to 'cuda' for training on a GPU.
  num_workers: 0 # 6  # num_workers >0 does not work on Mac, see https://stackoverflow.com/questions/64772335/pytorch-w-parallelnative-cpp206
  runtime_limits:
    max_time_per_run: 36000
    max_epochs_per_run: 30
  checkpoint_epochs: 15
"""
train_settings = yaml.safe_load(train_settings)
with open('02_training/train_settings.yaml', 'w') as outfile:
    yaml.dump(train_settings, outfile, default_flow_style=False)