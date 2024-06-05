import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(IMAGE_SIZE, NORM_MEAN, NORM_STD):
    both_transform = A.Compose(
        [
            A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
        ],
        additional_targets={"image0": "image"},
    )

    transform_only_input = A.Compose(
        [
            A.ColorJitter(p=0.1),
            A.Normalize(
                mean=[NORM_MEAN, NORM_MEAN, NORM_MEAN],
                std=[NORM_STD, NORM_STD, NORM_STD],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    transform_only_mask = A.Compose(
        [
            A.Normalize(
                mean=[NORM_MEAN, NORM_MEAN, NORM_MEAN],
                std=[NORM_STD, NORM_STD, NORM_STD],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    transforms = A.Compose(
        [
            A.Resize(width=256, height=256),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
            ToTensorV2(),
        ],
        additional_targets={"image0": "image"},
    )

    prediction_transform = A.Compose(
        [
            A.Resize(width=256, height=256),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
            ToTensorV2(),
        ],
    )

    return both_transform, transform_only_input, transform_only_mask, transforms, prediction_transform
