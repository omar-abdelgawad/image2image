"""Utility functions for the project."""

import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from torch import nn
from torch.nn import init

# from img2img import cfg


def prepare_sub_directories(path: str | Path) -> tuple[Path, Path]:
    """Creates subdirectories for the model training.

    Args:
        path (str | Path): usually ./out/model_dataset_name

    Returns:
        tuple[Path, Path]: weights_dir, eval_dir
    """
    # out
    # ├── model_dataset
    # │   ├── evaluation
    # │   └── last_trained_weights
    # └── model_dataset_2
    #     ├── evaluation
    #     └── last_trained_weights

    path = Path(path)
    eval_path = path / "evaluation"
    weights_path = path / "last_trained_weights"
    os.makedirs(eval_path, exist_ok=True)
    os.makedirs(weights_path, exist_ok=True)
    return weights_path, eval_path


def weights_init(
    init_type: str | Literal["gaussian", "xavier", "kaiming", "orthogonal", "default"],
) -> Callable[[nn.Module], None]:
    """Returns a function that Initializes weights for the model."""

    def init_fun(m: nn.Module) -> None:
        classname = m.__class__.__name__
        if (classname.find("Conv") == 0 or classname.find("Linear") == 0) and hasattr(
            m, "weight"
        ):
            if init_type == "gaussian":
                init.normal_(m.weight, 0.0, 0.02)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight, gain=math.sqrt(2))  # type: ignore
            elif init_type == "default":
                pass
            else:
                assert 0, f"Unsupported initialization: {init_type}"
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias, 0.0)

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
        assert 0, f"No correct model extension '.pt' found in {dirname}"
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


__all__ = ["prepare_sub_directories", "weights_init", "get_model_list"]
