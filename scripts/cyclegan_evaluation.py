from img2img.models.cyclegan.predictor import CycleGanPredictor
from img2img.data.cyclegan import HorseZebraDataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image


def main() -> int:
    monet_predictor = CycleGanPredictor(
        model_path="./out/saved_models/monet_training/genh.pth.tar"
    )
    yukiyoe_predictor = CycleGanPredictor(
        model_path="./out/saved_models/yukiyoe_training/genh.pth.tar"
    )
    vangogh_predictor = CycleGanPredictor(
        model_path="./out/saved_models/vangogh_training/genh.pth.tar"
    )
    dataset_path = "/media/omarabdelgawad/New Volume/Datasets/vangogh2photo/val"
    dataset = HorseZebraDataset(dataset_path + "/A", dataset_path + "/B")
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, pin_memory=True)
    for i, (painting, real_natural_view) in enumerate(data_loader):
        if i > 15:
            break
        img_grids = []
        # TODO: why is this a torch tensor?
        print(type(real_natural_view))
        batch_size = real_natural_view.size(0)

        for j in range(batch_size):
            real_natural_view_img = real_natural_view[j].numpy()

            # Get the predictions
            monet_img = monet_predictor(real_natural_view_img)
            vangogh_img = vangogh_predictor(real_natural_view_img)
            yukiyoe_img = yukiyoe_predictor(real_natural_view_img)

            # Concatenate images side by side
            img_grid = np.concatenate(
                (real_natural_view_img, monet_img, vangogh_img, yukiyoe_img), axis=1
            )
            img_grids.append(img_grid)

        # Concatenate all image grids vertically
        final_grid = np.concatenate(img_grids, axis=0)

        # Save the concatenated image
        img = Image.fromarray(final_grid)
        img.save(f"./saved_images/cycle_gan_monet_vangogh_yukiyoe{i}.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
