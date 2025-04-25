import numpy as np
from dingo.gw.prior import default_inference_parameters
from matplotlib import pyplot as plt

default_inference_parameters
# Load history.txt
filename = './history.txt'
data = np.loadtxt(filename, delimiter="\t")

# Plot loss values
plt.plot(data[:,0], data[:,1], label=f"training loss")
plt.plot(data[:,0], data[:,2], label=f"validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend();
plt.savefig("history.png")