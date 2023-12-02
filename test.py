import argparse
from typing import Optional
from typing import Sequence

def positive_float(value: str) -> float:
    ivalue = float(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is an invalid positive float value")
    return ivalue

def positive_int(value) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is an invalid positive int value")
    return ivalue

def custom_arg_parser(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="An unsupervised image-to-image translation model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-r", "--rate",
        type=positive_float,
        default=2e-4,
        help="Learning rate",
        dest="rate",
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=positive_int,
        default=16,
        help="Batch size",
        dest="batch_size",
    )
    parser.add_argument(
        "-w", "--num-workers",
        type=int,
        default=2,
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        help="Number of workers",
        dest="num_workers",
    )
    parser.add_argument(
        "-i", "--image-size",
        type=int,
        default=256,
        choices=[256, 512, 1024],
        help="Image size",
        dest="image_size",
    )
    parser.add_argument(
        "-e", "--num-epochs",
        type=positive_int,
        default=100,
        help="Number of epochs",
        dest="num_epochs",
    )
    parser.add_argument(
        "-l", "--load-model",
        action="store_true",
        help="Load model",
        dest="load_model",
    )
    parser.add_argument(
        "-n", "--no-save-model",
        action="store_false",
        help="Don't save the trained model",
        dest="save_model",
    )
    parser.add_argument(
        "-d", "--num-images-dataset",
        type=positive_int,
        default=1000,
        help="Number of images to use from the dataset",
        dest="num_images_dataset",
    )
    parser.add_argument(
        "-v", "--val-batch-size",
        type=positive_int,
        default=8,
        help="The number of images in the grid to show in the validation step during training",
        dest="val_batch_size",
    )

    args = parser.parse_args(argv)
    return args

if __name__ == "__main__":
    args = custom_arg_parser()
    print(args)