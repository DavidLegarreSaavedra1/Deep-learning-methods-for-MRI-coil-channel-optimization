from NNs.methods import *
import torch

def load_model(
    weights_path: str
):
    model = FastNN()

    model.load_state_dict(torch.load(weights_path))

    return model