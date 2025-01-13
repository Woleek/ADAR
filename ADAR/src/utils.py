# Copied from https://github.com/piotrkawa/audio-deepfake-source-tracing

import os
import random

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def adjust_learning_rate(args, lr, optimizer, epoch_num):
    lr = lr * (args.lr_decay ** (epoch_num // args.interval))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def mixup_data(x_mels, y, device, alpha=0.5):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x_mels.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x_mels = lam * x_mels + (1 - lam) * x_mels[index, :]
    y_a, y_b = y, y[index]
    return mixed_x_mels, y_a, y_b, lam


def regmix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
