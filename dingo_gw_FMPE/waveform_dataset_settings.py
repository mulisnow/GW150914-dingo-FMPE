import yaml

waveform_dataset_settings = """
domain:
  type: FrequencyDomain
  f_min: 20.0
  f_max: 256.0
  delta_f: 0.25 # Expressions like 1.0/8.0 would require eval and are not supported

waveform_generator:
  approximant: IMRPhenomPv2
  f_ref: 20.0
  spin_conversion_phase: 0.0   # Reference phase when converting from spin angles to Cartesian spins. If None, use phase parameter.
  # f_start: 15.0  # Optional setting useful for EOB waveforms. Overrides f_min when generating waveforms.

# Dataset only samples over intrinsic parameters. Extrinsic parameters are chosen at train time.
intrinsic_prior:
  mass_1: bilby.core.prior.Constraint(minimum=20.0, maximum=40.0)
  mass_2: bilby.core.prior.Constraint(minimum=20.0, maximum=40.0)
  chirp_mass: bilby.gw.prior.UniformInComponentsChirpMass(minimum=15.0, maximum=50.0)
  mass_ratio: bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.125, maximum=1.0)
  theta_jn: 2.624497413635254
  tilt_1: 2.0111560821533203
  tilt_2: 1.0743615627288818
  a_1: 0.925635814666748
  a_2: 0.5538952350616455
  phi_jl: 5.561878204345703
  phase: 0.9604566579018894
  # Reference values for fixed (extrinsic) parameters. These are needed to generate a waveform.
  luminosity_distance: 100.0  # Mpc
  geocent_time: 0.0 # s

# Dataset size
num_samples: 50_000

compression: None
"""
waveform_dataset_settings = yaml.safe_load(waveform_dataset_settings)
with open('01_training_data/waveform_dataset/waveform_dataset_settings.yaml', 'w') as outfile:
    yaml.dump(waveform_dataset_settings, outfile, default_flow_style=False)