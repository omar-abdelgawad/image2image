"""Main script for training the model. Can train from scratch or resume from a checkpoint."""
from img2img import cfg
from img2img.models.pix2pix.trainer import Pix2PixTrainer


def main() -> int:
    """Entry point for training loop."""
    path = cfg.OUT_PATH / f"pix2pix_{cfg.CHOSEN_DATASET.value.stem}"
    trainer = Pix2PixTrainer(
        load_model=cfg.LOAD_MODEL,
        learning_rate=cfg.LEARNING_RATE,
        betas=cfg.BETA_OPTIM,
        train_batch_size=cfg.BATCH_SIZE,
        val_batch_size=cfg.VAL_BATCH_SIZE,
        device=cfg.DEVICE,
        path=path,
        num_workers=cfg.NUM_WORKERS,
        train_dataset_path=cfg.TRAIN_DATASET_PATH,
        val_dataset_path=cfg.VAL_DATASET_PATH,
        chosen_dataset=cfg.CHOSEN_DATASET,
        l1_lambda=cfg.L_1_LAMBDA,
    )
    trainer.train(
        num_epochs=cfg.NUM_EPOCHS,
        save_model=cfg.SAVE_MODEL,
        checkpoint_period=cfg.CHECKPOINT_PERIOD,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
