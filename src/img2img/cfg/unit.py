from .pix2pix import *

# unit config
WEIGHT_DECAY = 0.0001
LR_POLICY = "step"
STEP_SIZE = 100000
GAMMA = 0.5
INIT = "kaiming"
GAN_WEIGHT = 1
RECONSTRUCTION_X_WEIGHT = 10
RECONSTRUCTION_H_WEIGHT = 0
RECONSTRUCTION_KL_WEIGHT = 0.01
RECONSTRUCTION_X_CYC_WEIGHT = 10
RECONSTRUCTION_KL_CYC_WEIGHT = 0.01

"""unit transforms

# TODO: understand the augmentations below and improve them (maybe add more augmentations).
both_transform = A.Compose(
    [
        A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
    ],
    additional_targets={"image0": "image"},
)

# this is equivalent to first domain transform
transform_only_input = A.Compose(
    [
        A.ColorJitter(p=0.1),
        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            max_pixel_value=255.0,
        ),
        A.HorizontalFlip(p=0.5),
        ToTensorV2(),
    ]
)

# this is equivalent to second domain transform
transform_only_mask = A.Compose(
    [
        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)
"""

# TODO: make transforms differ from model to another
