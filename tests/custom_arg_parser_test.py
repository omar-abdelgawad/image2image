import model

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
args = model.custom_arg_parser([])

def test_arg_parser_learning_rate():
    assert args.lr == 2e-4

def test_arg_parser_batch_size():
    assert args.b == 16

def test_arg_parser_num_workers():
    assert args.w == 2

def test_arg_parser_image_size():
    assert args.i == 256

def test_arg_parser_num_epochs():
    assert args.e == 100

def test_arg_parser_load_model():
    assert args.l == True

def test_arg_parser_save_model():
    assert args.s == False

def test_arg_parser_num_images_dataset():
    assert args.d == 1000

def test_arg_parser_val_batch_size():
    assert args.v == 8
