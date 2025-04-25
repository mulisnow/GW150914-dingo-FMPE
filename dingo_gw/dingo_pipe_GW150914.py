dingo_pipe_GW150914 = """
################################################################################
##  Job submission arguments
################################################################################

local = True
submit = False
accounting = dingo
request-cpus-importance-sampling = 2
simple-submission = False

################################################################################
##  Sampler arguments
################################################################################

model = 02_training/model_latest.pt
device = cpu
num-samples = 5000
batch-size = 5000
recover-log-prob = true
importance-sampling-settings = {}

################################################################################
## Data generation arguments
################################################################################

trigger-time = 1126259462.3885767 # GW150914 # condition network on maxlog geocent time 1126259462.4 - 0.011423417367041111
label = GW150914
outdir = 03_inference/outdir_GW150914
channel-dict = {H1:GWOSC, L1:GWOSC}
psd-length = 128
# sampling-frequency = 2048.0
# importance-sampling-updates = {'duration': 4.0}

################################################################################
## Plotting arguments
################################################################################

plot-corner = true
plot-weights = true
plot-log-probs = true
"""
with open('03_inference/GW150914.ini', 'w') as outfile:
    outfile.write(dingo_pipe_GW150914)