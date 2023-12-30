"""Test module for custom_arg_parser function"""
from pix2pix.cli import custom_arg_parser


# TODO: use pytest.mark.parametrize instead of multiple tests
# @pytest.mark.parametrize(
#         ('input_user', 'expected'),
#         (
#             ('lr', 2e-4),
#             ('b', 16),
#             ('w', 2),
#             ('i', 256),
#             ('e', 100),
#             ('l', False),
#             ('s', True),
#             ('d', 1000),
#         ),
# )
def test_default_arguments():
    """Test default arguments"""
    args = custom_arg_parser([])

    assert args.rate == 2e-4
    assert args.batch_size == 16
    assert args.num_workers == 2
    assert args.image_size == 256
    assert args.num_epochs == 100
    assert args.load_model is False
    assert args.save_model is True
    assert args.num_images_dataset == 1000
    assert args.val_batch_size == 8
