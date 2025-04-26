import numpy as np
from dingo.gw.dataset.waveform_dataset import WaveformDataset
from matplotlib import pyplot as plt

# Load dataset
waveform_dataset_path = '01_training_data/waveform_dataset/waveform_dataset.hdf5'
wfd = WaveformDataset(file_name=waveform_dataset_path)
print("One Datapoint contains:", wfd[0].keys())
print("Parameters contains: ", wfd[0]['parameters'].keys())
print("Waveform contains:", wfd[0]['waveform'].keys())
# Plot an exemplary waveform
f_domain = wfd.domain.sample_frequencies
data_sample = wfd[0]['waveform']

plt.plot(f_domain, data_sample['h_cross'].real, c="blue", label=r'$h_{\times}$ real')
plt.plot(f_domain, data_sample['h_cross'].imag, c="cornflowerblue", label=r'$h_{\times}$ imag')
plt.plot(f_domain, data_sample['h_plus'].real, c="tab:red", label=r'$h_+$ real')
plt.plot(f_domain, data_sample['h_plus'].imag, c="coral", label=r'$h_+$ imag')
plt.xlabel(r"Frequency $f$ [Hz]")
plt.ylabel(r"Polarization $h$")
plt.legend();

# Convert dictionary into numpy array
data_sample_np = np.array([data_sample['h_cross'], data_sample['h_plus']])

# Time-translate sample
time_translated_sample = wfd.domain.time_translate_data(data_sample_np, dt=0.1)

# Plot
fig, axs = plt.subplots(1, 2, sharey=True, figsize=(8,3.5))
axs[0].plot(f_domain, data_sample['h_cross'].real, c='blue', label=r'$h_{\times}$ real')
axs[0].plot(f_domain, data_sample['h_plus'].real, c='tab:red', label=r'$h_+$ real')
axs[0].set_xlabel(r"Frequency $f$ [Hz]")
axs[0].set_ylabel(r"Polarization $h$")
axs[0].legend()

axs[1].plot(f_domain, time_translated_sample[0].real, c="blue", label=r'Time-translated $h_{\times}$ real')
axs[1].plot(f_domain, time_translated_sample[1].real, c="tab:red", label=r'Time-translated $h_+$ real')
axs[1].set_xlabel(r"Frequency $f$ [Hz]")
axs[1].legend();
plt.savefig('01_training_data/result/waveform_dataset_darw.png')