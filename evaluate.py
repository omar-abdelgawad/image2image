import torch
from torch.utils.data import DataLoader

from model import cfg
from model.generator import Generator
from model.dataset import create_dataset
from model.utils import evaluate_val_set


def main() -> int:
    """Evaluates the model on validation set."""
    gen = Generator(in_channels=3).to(cfg.DEVICE)
    print(f"Loading model from {cfg.CHECKPOINT_GEN}")
    checkpoint = torch.load(cfg.CHECKPOINT_GEN, map_location=cfg.DEVICE)
    gen.load_state_dict(checkpoint["state_dict"])
    print("Model loaded successfully! Creating validation set...")
    val_dataset = create_dataset(
        root_dir=cfg.VAL_DATASET_PATH, dataset_type=cfg.CHOSEN_DATASET
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg.VAL_BATCH_SIZE, shuffle=False)
    print("Validation set created successfully! Evaluating...")
    evaluate_val_set(gen, val_loader, cfg.EVALUATION_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
