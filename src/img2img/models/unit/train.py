"""Train script for UNIT model."""

from tqdm import tqdm


from img2img import cfg
from img2img.data import get_loader
from img2img.utils import prepare_sub_directories
from img2img.models.unit.utils import save_some_examples

from img2img.models.unit.trainer import UNIT_Trainer


def trainfn(trainer, train_loader):
    loop = tqdm(train_loader, leave=True)
    for idx, (images_a, images_b) in enumerate(loop):
        images_a, images_b = images_a.to(cfg.DEVICE), images_b.to(cfg.DEVICE)
        # trainer.update_learning_rate()

        trainer.dis_update(images_a, images_b)
        trainer.gen_update(images_a, images_b)
        trainer.update_learning_rate()


def main() -> int:
    """Entry point."""
    trainer = UNIT_Trainer()
    trainer.to(cfg.DEVICE)
    train_loader = get_loader(
        root_dir=cfg.TRAIN_DATASET_PATH,
        dataset_type=cfg.CHOSEN_DATASET,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
    )
    val_loader = get_loader(
        root_dir=cfg.VAL_DATASET_PATH,
        dataset_type=cfg.CHOSEN_DATASET,
        batch_size=cfg.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )
    path = cfg.OUT_PATH / f"unit_{cfg.CHOSEN_DATASET.value.stem}"
    weights_dir, val_dir = prepare_sub_directories(path)
    # TODO: copy the config yaml file to the out directory
    # shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder
    # TODO: apply mixed precision (torch.cuda.amp.autocast)
    if cfg.LOAD_MODEL:
        trainer.resume(weights_dir)
    for epoch in range(cfg.NUM_EPOCHS):
        print(f"Epoch: {epoch}")
        trainfn(trainer=trainer, train_loader=train_loader)
        save_some_examples(
            trainer=trainer,
            val_loader=val_loader,
            epoch=epoch,
            dir_path=val_dir,
        )
        if cfg.SAVE_MODEL and epoch % 5 == 0:
            trainer.save(weights_dir, epoch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
