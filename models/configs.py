
class DefaultConfig(object):
    model = 'ResNet'
    load_model_path = None
    use_gpu = True
    num_workers = 2
    print_freq = 10

    train_data_path = "data/train/train-images.gz"
    train_label_path = "data/train/train-labels.gz"
    test_data_path = "data/test/test-images.gz"
    test_label_path = "data/test/test-label.gz"

    image_size = 28
    pixel_depth = 255
    train_image_nums = 60000
    test_image_nums = 10000

    seed = 10
    batch_size = 64
    epoch = 10
    lr = 0.001


