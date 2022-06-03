from NNs.methods import *
import torch

def load_model(
    weights_path: str,
    image_size: int,
):
    model = FastNN(image_size)

    model.load_state_dict(torch.load(weights_path))

    return model
