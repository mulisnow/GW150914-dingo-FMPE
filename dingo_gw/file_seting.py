import os

os.makedirs('01_training_data/asd_dataset', exist_ok=True)
os.makedirs('01_training_data/waveform_dataset', exist_ok=True)
os.makedirs('02_training', exist_ok=True)
os.makedirs('03_inference/injection', exist_ok=True)
os.makedirs('04_exercise/lum_dist_marginalization', exist_ok=True)
os.makedirs('04_exercise/with_lum_dist', exist_ok=True)
os.makedirs('05_pretrained_model/init_train_dir', exist_ok=True)
os.makedirs('05_pretrained_model/main_train_dir', exist_ok=True)