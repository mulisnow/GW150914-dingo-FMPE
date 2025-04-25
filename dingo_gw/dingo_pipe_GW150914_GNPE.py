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

model-init = 05_pretrained_model/init_train_dir/model_init.pt
model = 05_pretrained_model/main_train_dir/model.pt
device = cpu
num-gnpe-iterations = 5 #30
num-samples = 5000
batch-size = 5000
recover-log-prob = False
prior-dict = {
luminosity_distance = bilby.gw.prior.UniformComovingVolume(minimum=100, maximum=2000, name='luminosity_distance'),
}
importance-sample = False

################################################################################
## Data generation arguments
################################################################################

trigger-time = GW150914
label = GW150914
outdir = 05_pretrained_model/outdir_GW150914
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
with open('05_pretrained_model/GW150914.ini', 'w') as outfile:
    outfile.write(dingo_pipe_GW150914)