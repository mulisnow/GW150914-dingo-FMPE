from dingo.gw.noise.asd_dataset import ASDDataset
from matplotlib import pyplot as plt

from dingo_gw.waveform_dataset_darw import f_domain, wfd

# Load ASD dataset
asd_dataset_path = '01_training_data/asd_dataset/asd_GW150914.hdf5'
asds = ASDDataset(file_name=asd_dataset_path, domain_update=wfd.domain.domain_dict)
# Get ASD sample
asd_sample = asds.sample_random_asds()
print("One Datapoint contains:", asd_sample.keys())

plt.loglog(f_domain, asd_sample['H1'], label=r'H1')
plt.loglog(f_domain, asd_sample['L1'], label=r'L1')
plt.xlabel(r"Frequency [Hz]")
plt.ylabel(r"Noise ASD $[1/\sqrt{\mathrm{Hz}}]$")
plt.xlim([wfd.domain.f_min, wfd.domain.f_max])
plt.ylim([1.e-24, 1.e-20])
plt.legend();
plt.savefig('01_training_data/result/waveform_dat')
