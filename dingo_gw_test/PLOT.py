from dingo.gw.result import Result
import matplotlib.pyplot as plt

# 加载结果
result = Result.from_file("03_inference/injection/result.hdf5")
samples = result.samples

# 绘制Corner图（真实值：arXiv:1502.07715）
result.plot_corner(
    parameters=["chirp_mass", "mass_ratio"],
    truths={"chirp_mass": 31.2, "mass_ratio": 0.85},
    ranges={
        "chirp_mass": [30, 32],
        "mass_ratio": [0.7, 1.0]
    },
    filename="corner.png"
)
plt.show()