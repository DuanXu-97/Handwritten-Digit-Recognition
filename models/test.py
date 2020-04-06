import os
import argparse
import torch as t
from torch.utils.data import DataLoader
from torchnet import meter

import models
from data.dataset import MNIST
from models import configs


def test(config):
    opt.parse(kwargs)

    # configure model
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)

    # data
    test_data = MNIST(data_path=config.test_data_path, label_path=config.test_label_path, config=config, mode='test')
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    results = []
    for ii, (input, path) in tqdm(enumerate(test_dataloader)):
        if opt.use_gpu:
            input = input.cuda()
        score = model(input)
        probability = t.nn.functional.softmax(score, dim=1)[:, 0].detach().tolist()
        # label = score.max(dim = 1)[1].detach().tolist()

        batch_results = [(path_.item(), probability_) for path_, probability_ in zip(path, probability)]

        results += batch_results
    # write_csv(results, opt.result_file)

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='ResNet', help="model to be used")
    parser.add_argument('--use_gpu', action='store_true', help="whether use gpu")
    parser.add_argument('--load_model_path', type=str, default=None, help="Path of pre-trained model")
    parser.add_argument('--ckpts_dir', type=str, default=None, help="Dir to store checkpoints")

    args = parser.parse_args()
    config = configs.DefaultConfig()

    if not os.path.exists(args.ckpts_dir):
        os.makedirs(args.ckpts_dir)

    train(args, config)