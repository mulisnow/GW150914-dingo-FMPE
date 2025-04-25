import torch
from dingo.core.posterior_models import NormalizingFlowPosteriorModel

def load_model(path):
    try:
        state_dict = torch.load(path, map_location="cpu")
        print(f"{path} 加载成功！Keys:", state_dict.keys())
        model = NormalizingFlowPosteriorModel(
            model_filename=path,
            device="cpu",
            map_location="cpu"
        )
        print("模型完整加载验证通过")
    except Exception as e:
        print(f"加载失败: {str(e)}")

load_model("05_pretrained_model/init_train_dir/model_init.pt")
load_model("05_pretrained_model/main_train_dir/model.pt")