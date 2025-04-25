import pickle

import yaml

asd_dataset_settings = """
dataset_settings:
#  f_min: 0         # defaults to 0
#  f_max: 2048      # defaults to f_s/2
  f_s: 4096
  time_psd: 1024
  T: 2.0
  window:
    roll_off: 0.4
    type: tukey
  time_gap: 0 # specifies the time skipped between to consecutive PSD estimates. If set < 0, the time segments overlap
  num_psds_max: 1 # if set > 0, only a subset of all available PSDs will be used
  detectors:
    - H1
    - L1
  observing_run: O1
"""
asd_dataset_settings = yaml.safe_load(asd_dataset_settings)
with open('01_training_data/asd_dataset/asd_dataset_settings.yaml', 'w') as outfile:
    yaml.dump(asd_dataset_settings, outfile, default_flow_style=False)
time_GW150914 = 1126259462.391 - 0.0114 # maxlog geocent time of GW150914
asd_start_time = time_GW150914 - asd_dataset_settings["dataset_settings"]["T"]
asd_end_time = time_GW150914
time_segments = {
    'H1': [[asd_start_time, asd_end_time]],
    'L1': [[asd_start_time, asd_end_time]]
}

with open('01_training_data/asd_dataset/time_segment_GW150914.pkl', 'wb') as f:
    pickle.dump(time_segments, f)