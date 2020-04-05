import gzip
import numpy as np
from torch.utils import data
import torchvision.transforms as T


class MNIST(data.Dataset):

    def __init__(self, data_path, label_path, config, mode):

        self.config = config
        self.mode = mode

        if mode == 'test':
            self.image_nums = config.test_image_nums
        elif mode == 'train' or mode == 'valid':
            self.image_nums = config.train_image_nums
        else:
            raise Exception('Error Mode.')

        with gzip.open(data_path) as bytestream:
            bytestream.read(16)
            buf = bytestream.read(config.image_size * config.image_size * self.image_nums * config.num_channels)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            data = (data - (config.pixel_depth / 2.0)) / config.pixel_depth
            self.data = data.reshape(self.image_nums, config.image_size, config.image_size, config.num_channels)

        with gzip.open(label_path) as bytestream:
            bytestream.read(8)
            buf = bytestream.read(1 * self.image_nums)
            self.labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

        if mode == 'train':
            self.image_nums = config.train_image_nums * 0.7
            self.data = self.data[:int(config.train_image_nums * 0.7)]
            self.labels = self.labels[:int(config.train_image_nums * 0.7)]
        elif mode == 'valid':
            self.image_nums = config.train_image_nums - config.train_image_nums * 0.7
            self.data = self.data[int(config.train_image_nums * 0.7):]
            self.labels = self.labels[int(config.train_image_nums * 0.7):]

    def __getitem__(self, index):
        return T.ToTensor()(self.data[index]), self.labels[index]

    def __len__(self):
        return self.data.shape[0]


