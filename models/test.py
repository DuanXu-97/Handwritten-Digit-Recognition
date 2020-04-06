import sys
sys.path.append('../')
import os
import argparse
import torch as t
from torch.utils.data import DataLoader
from torchnet import meter
from sklearn.metrics import precision_score, recall_score, f1_score

from models import network
from data.dataset import MNIST
from models import configs


def test(args, config):

    test_set = MNIST(data_path=config.test_data_path, label_path=config.test_label_path, config=config, mode='test')
    test_dataloader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = getattr(network, args.model)().eval()
    if args.load_model_path:
        model.load(args.load_model_path)
    if args.use_gpu:
        model.cuda()

    y_true = []
    y_pred = []
    test_confusion_matrix = meter.ConfusionMeter(10)
    test_confusion_matrix.reset()

    model.eval()
    for _iter, (test_data, test_label) in enumerate(test_dataloader):

        if args.use_gpu:
            test_data = test_data.cuda()

        test_logits, test_output = model(test_data)
        y_true.extend(test_label.numpy().tolist())
        y_pred.extend(test_logits.max(dim=1)[1].detach().tolist())
        test_confusion_matrix.add(test_logits.detach().squeeze(), test_label.type(t.LongTensor))

    print(y_true[:100])
    print(y_pred[:100])

    test_cm = test_confusion_matrix.value()
    test_metrics = dict()
    test_metrics['accuracy'] = 100. * (test_cm.diagonal().sum()) / (test_cm.sum())
    test_metrics['precision'], test_metrics['recall'], test_metrics['f1'] = dict(), dict(), dict()
    test_metrics['precision']['micro'] = precision_score(y_true, y_pred, average='micro')
    test_metrics['precision']['macro'] = precision_score(y_true, y_pred, average='macro')
    test_metrics['recall']['micro'] = recall_score(y_true, y_pred, average='micro')
    test_metrics['recall']['macro'] = recall_score(y_true, y_pred, average='macro')
    test_metrics['f1']['micro'] = f1_score(y_true, y_pred, average='micro')
    test_metrics['f1']['macro'] = f1_score(y_true, y_pred, average='macro')

    print("test_metrics:", test_metrics)
    print("test_cm:\n{test_cm}".format(
        test_cm=str(test_cm),
    ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='ResNet', help="model to be used")
    parser.add_argument('--use_gpu', action='store_true', help="whether use gpu")
    parser.add_argument('--load_model_path', type=str, required=True, help="Path of trained model")

    args = parser.parse_args()
    config = configs.DefaultConfig()

    test(args, config)
