import sys
sys.path.append('../')
import os
import time
import argparse
import torch as t
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchnet import meter
from models import configs
from models import network
from data.dataset import MNIST
from utils.visualize import Visualizer


def train(args, config):
    vis = Visualizer()

    train_data = MNIST(data_path=config.train_data_path, label_path=config.train_label_path, config=config, mode='train')
    valid_data = MNIST(data_path=config.train_data_path, label_path=config.train_label_path, config=config, mode='valid')

    train_dataloader = DataLoader(train_data, config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers)
    valid_dataloader = DataLoader(valid_data, config.batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers)

    model = getattr(network, args.model)().eval()
    if args.load_model_path:
        model.load(args.load_model_path)
    if args.use_gpu:
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    train_loss_meter, valid_loss_meter = meter.AverageValueMeter(), meter.AverageValueMeter()
    train_confusion_matrix, valid_confusion_matrix = meter.ConfusionMeter(10), meter.ConfusionMeter(10)

    best_valid_loss = 0
    best_epoch = 0
    dist_to_best = 4

    time_begin = time.clock()

    for epoch in range(config.epoch):

        # train
        model.train()
        train_loss_meter.reset()
        train_confusion_matrix.reset()

        for iter, (train_data, train_target) in enumerate(train_dataloader):

            if args.use_gpu:
                train_data = train_data.cuda()
                train_target = train_target.cuda()

            optimizer.zero_grad()
            train_logits, train_output = model(train_data)
            train_loss = criterion(train_logits, train_target)
            train_loss.backward()
            optimizer.step()

            train_loss_meter.add(train_loss.item())
            train_confusion_matrix.add(train_logits.data, train_target.data)

            if iter % config.print_freq == 0:
                vis.plot('train_loss', train_loss_meter.value()[0])
        model.save(path=os.path.join(args.ckpts_dir, 'model_{0}.pth'.format(str(epoch))))

        # valid
        model.eval()
        valid_loss_meter.reset()
        valid_confusion_matrix.reset()

        for iter, (valid_data, valid_target) in enumerate(valid_dataloader):

            if args.use_gpu:
                valid_data = valid_data.cuda()
                valid_target = valid_target.cuda()

            valid_logits, valid_output = model(valid_data)
            valid_loss = criterion(valid_logits, valid_target)

            valid_loss_meter.add(valid_loss.item())
            valid_confusion_matrix.add(valid_logits.detach().squeeze(), valid_target.type(t.LongTensor))

        valid_cm = valid_confusion_matrix.value()
        valid_accuracy = 100. * (valid_cm.diagonal().sum()) / (valid_cm.sum())
        vis.plot('valid_accuracy', valid_accuracy)

        vis.log("epoch:{epoch}, lr:{lr}, train_loss:{train_loss}, train_cm:{train_cm}, valid_loss:{valid_loss}, valid_cm:{valid_cm}".format(
            epoch=epoch,
            train_loss=train_loss_meter.value()[0],
            train_cm=str(train_confusion_matrix.value()),
            valid_loss=valid_loss_meter.value()[0],
            valid_cm=str(valid_cm),
            lr=config.lr
        ))

        # early stop
        if valid_loss_meter.value()[0] < best_valid_loss:
            best_epoch = epoch
            best_valid_loss = valid_loss_meter.value()[0]
            dist_to_best = 0

        dist_to_best += 1
        if dist_to_best > 4:
            break

    model.save(path=os.path.join(args.ckpts_dir, 'model.pth'))
    vis.save()
    print("save model successfully")
    print("best epoch: ", best_epoch)
    print("best valid loss: ", best_valid_loss)
    time_end = time.clock()
    print('time cost: %.2f' % (time_end - time_begin))


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


