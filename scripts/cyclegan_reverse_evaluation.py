from img2img.models.cyclegan.predictor import CycleGanPredictor
from img2img.data.cyclegan import HorseZebraDataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import torch


def main() -> int:
    monet_predictor = CycleGanPredictor(
        model_path="./out/cyclegan_monet2photo/last_trained_weights/genz.pth.tar"
    )
    vangogh_predictor = CycleGanPredictor(
        model_path="./out/cyclegan_vangogh2photo/last_trained_weights/genz.pth.tar"
    )
    yukiyoe_predictor = CycleGanPredictor(
        model_path="./out/cyclegan_yukiyoe2photo/last_trained_weights/genz.pth.tar"
    )
    monet_dataset_path = "/media/omarabdelgawad/New Volume/Datasets/monet2photo/val"
    vanghogh_dataset_path = (
        "/media/omarabdelgawad/New Volume/Datasets/vangogh2photo/val"
    )
    yukiyoe_dataset_path = "/media/omarabdelgawad/New Volume/Datasets/ukiyoe2photo/val"
    monet_dataset = HorseZebraDataset(
        monet_dataset_path + "/A", monet_dataset_path + "/B"
    )
    monet_data_loader = DataLoader(
        monet_dataset, batch_size=4, shuffle=True, pin_memory=True
    )
    for i, (painting, real_natural_view) in enumerate(monet_data_loader):
        if i > 10:
            break
        img_grids = []
        batch_size = real_natural_view.size(0)
        for j in range(batch_size):
            painting_image = painting[j].numpy()
            real_monet_view = monet_predictor(painting_image)
            # save
            img_grid = np.concatenate((painting_image, real_monet_view), axis=1)
            img_grids.append(img_grid)
        final_grid = np.concatenate(img_grids, axis=0)
        img = Image.fromarray(final_grid)
        img.save(f"./saved_images/cycle_gan_monet_{i}.jpg")

    vangogh_dataset = HorseZebraDataset(
        vanghogh_dataset_path + "/A", vanghogh_dataset_path + "/B"
    )
    vangogh_data_loader = DataLoader(
        vangogh_dataset, batch_size=4, shuffle=True, pin_memory=True
    )
    for i, (painting, real_natural_view) in enumerate(vangogh_data_loader):
        if i > 10:
            break
        img_grids = []
        batch_size = real_natural_view.size(0)
        for j in range(batch_size):
            painting_image = painting[j].numpy()
            real_vangogh_view = vangogh_predictor(painting_image)
            # save
            img_grid = np.concatenate((painting_image, real_vangogh_view), axis=1)
            img_grids.append(img_grid)
        final_grid = np.concatenate(img_grids, axis=0)
        img = Image.fromarray(final_grid)
        img.save(f"./saved_images/cycle_gan_vangogh_{i}.jpg")

    yukiyoe_dataset = HorseZebraDataset(
        yukiyoe_dataset_path + "/A", yukiyoe_dataset_path + "/B"
    )
    yukiyoe_data_loader = DataLoader(
        yukiyoe_dataset, batch_size=4, shuffle=True, pin_memory=True
    )
    for i, (painting, real_natural_view) in enumerate(yukiyoe_data_loader):
        if i > 10:
            break
        img_grids = []
        batch_size = real_natural_view.size(0)
        for j in range(batch_size):
            painting_image = painting[j].numpy()
            real_yukiyoe_view = yukiyoe_predictor(painting_image)
            # save
            img_grid = np.concatenate((painting_image, real_yukiyoe_view), axis=1)
            img_grids.append(img_grid)
        final_grid = np.concatenate(img_grids, axis=0)
        img = Image.fromarray(final_grid)
        img.save(f"./saved_images/cycle_gan_yukiyoe_{i}.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
