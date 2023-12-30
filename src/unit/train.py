import os

from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

# from torch.backends.cudnn import benchmark

from unit import cfg
from unit.utils import (
    get_data_loaders,
    prepare_sub_directories,
    save_some_examples,
)
from unit.trainer import UNIT_Trainer

_WRITER = SummaryWriter("runs/expirement_1")


def trainfn(trainer, train_loader):
    loop = tqdm(train_loader, leave=True)
    for idx, (images_a, images_b) in enumerate(loop):
        images_a, images_b = images_a.to(cfg.DEVICE), images_b.to(cfg.DEVICE)
        trainer.update_learning_rate()

        trainer.dis_update(images_a, images_b)
        trainer.gen_update(images_a, images_b)
        trainer.update_learning_rate()


def main() -> int:
    trainer = UNIT_Trainer()
    trainer.to(cfg.DEVICE)
    train_loader, val_loader = get_data_loaders()
    model_name_from_dataset = "UNIT_" + cfg.CHOSEN_DATASET.value.stem
    prepare_sub_directories("./out")
    # TODO: copy the config yaml file to the out directory
    # shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder
    # TODO: apply mixed precision (torch.cuda.amp.autocast)
    if cfg.LOAD_MODEL:
        trainer.resume(cfg.CHECKPOINT_DIR)
    for epoch in range(cfg.NUM_EPOCHS):
        print(f"Epoch: {epoch}")
        save_some_examples(
            trainer, val_loader, epoch, folder=cfg.EVALUATION_PATH, writer=_WRITER
        )
        trainfn(trainer=trainer, train_loader=train_loader)
        if cfg.SAVE_MODEL and epoch % 5 == 0:
            trainer.save(cfg.CHECKPOINT_DIR, epoch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
