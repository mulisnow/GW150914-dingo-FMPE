from dingo.core.posterior_models import FlowMatchingPosteriorModel
from dingo.core.posterior_models.normalizing_flow import NormalizingFlowPosteriorModel
from dingo.gw.inference.gw_samplers import GWSampler
from dingo.gw.injection import Injection
from dingo.gw.noise.asd_dataset import ASDDataset

model_path = "02_training_FMPE/model_latest.pt"
asd_path = "01_training_data/asd_dataset/asd_GW150914.hdf5"

# Load the network into the GWSampler class
pm = FlowMatchingPosteriorModel(model_filename=model_path, device="cpu")
sampler = GWSampler(model=pm)

# Generate an injection consistent with the data the model was trained on.
injection = Injection.from_posterior_model_metadata(pm.metadata)
injection.asd = ASDDataset(asd_path, ifos=["H1", "L1"])
theta = injection.prior.sample()
inj = injection.injection(theta)

# Generate 10,000 samples from the DINGO model based on the generated injection data.
sampler.context = inj
sampler.run_sampler(10_000)
result = sampler.to_result()

# The following are only needed for importance-sampling the result.
result.importance_sample(num_processes=8)

# Make a corner plot and save the result.
result.print_summary()
kwargs = {"legend_font_size": 15, "truth_color": "black"}
result.plot_corner(parameters=["chirp_mass", "mass_ratio"],
                   filename="03_inference/injection/corner.pdf",
                   truths=theta,
                   **kwargs)
result.to_file("03_inference/injection/result.hdf5")