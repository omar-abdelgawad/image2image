import os
import math
from pathlib import Path
from typing import Callable

from torch import nn
from torch.nn import init


def prepare_sub_directories(path: str | Path) -> None:
    """Creates subdirectories for saving images and checkpoints.
    Creates the directory given +
        evaluation subdirectory
        last_trained_weights subdirectory
        saved_models subdirectory
    """
    path = Path(path)
    os.makedirs(path, exist_ok=True)
    os.makedirs(path / "evaluation", exist_ok=True)
    os.makedirs(path / "last_trained_weights", exist_ok=True)
    os.makedirs(path / "saved_models", exist_ok=True)


def weights_init(init_type: str = "gaussian") -> Callable[[nn.Module], None]:
    """Returns a function that Initializes weights for the model."""

    def init_fun(m: nn.Module) -> None:
        classname = m.__class__.__name__
        if (classname.find("Conv") == 0 or classname.find("Linear") == 0) and hasattr(
            m, "weight"
        ):
            # print m.__class__.__name__
            if init_type == "gaussian":
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))  # type: ignore
            elif init_type == "default":
                pass
            else:
                assert 0, f"Unsupported initialization: {init_type}"
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


# Get model list for resume
def get_model_list(dirname, key):
    """Returns the latest model_weights(.pt) from a directory."""
    if os.path.exists(dirname) is False:
        raise ValueError(f"{dirname} does not exist")
    gen_models = [
        os.path.join(dirname, f)
        for f in os.listdir(dirname)
        if os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f
    ]
    if not gen_models:
        raise ValueError(f"No correct model extension '.pt' found in {dirname}")
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


__all__ = ["prepare_sub_directories", "weights_init", "get_model_list"]
