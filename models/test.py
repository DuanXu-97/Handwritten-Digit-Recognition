import torch as t
from torch.utils.data import DataLoader
from torchnet import meter

import models
from data.dataset import Minst
from models.configs import opt

def test(**kwargs):
    """
    测试
    :param kwargs:
    :return:
    """
    opt.parse(kwargs)

    # configure model
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)

    # data
    train_data = Minst(data_root=opt.test_image_path, test=True)
    test_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
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