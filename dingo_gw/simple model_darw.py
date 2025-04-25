import numpy as np
from matplotlib import pyplot as plt

filename = '02_training/history.txt'
data = np.loadtxt(filename, delimiter="\t")

# Plot loss values
plt.plot(data[:,0], data[:,1], label=f"training loss")
plt.plot(data[:,0], data[:,2], label=f"validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend();
plt.savefig('02_training/result/loss')