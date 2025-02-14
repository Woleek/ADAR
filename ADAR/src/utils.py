# Copied from https://github.com/piotrkawa/audio-deepfake-source-tracing

import json
import os
import random
from typing import Any, Dict

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
    
    
def load_superclass_mapping(dataset_path, superclass_mapping_path):
    id_map = {}
    name_map = {}
    with open(os.path.join(dataset_path, superclass_mapping_path), "r") as f:
        for line in f.readlines()[1:]:
            model, superclass, label, suplabel = line.replace('\n', '').split(',')
            id_map[int(label)] = int(suplabel)
            name_map[model] = superclass
            
    # Count unique superclasses
    num_sup = len(set(id_map.values()))
    next_new_id = num_sup - 1
    
    # assign new superclass IDs to models labeled with -1 (unknown superclass)
    for sub, sup in id_map.items():
        if sup == -1:
            id_map[sub] = next_new_id
            # id_map[sub] = num_sup - 1
            next_new_id += 1
        
    return id_map, name_map


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


def save_checkpoint(
    save_folder, 
    model_state, 
    optimizer_state, 
    center_loss_optimizer_state=None, 
    training_stats=None, 
    epoch=None
):
    # Save model
    if epoch is not None:
        chpt_path = os.path.join(save_folder, "checkpoint", f"anti-spoofing_feat_model_{epoch:02d}.pth")
        print(f"[INFO] Saving intermediate checkpoint to {chpt_path}")
    else:
        chpt_path = os.path.join(save_folder, "anti-spoofing_feat_model.pth")
        print(f"[INFO] Saving model to {chpt_path}")
    
    torch.save(model_state, chpt_path)
    
    # Save optimizer
    if epoch is not None:
        opt_path = os.path.join(save_folder, "checkpoint", f"optimizer_{epoch:02d}.pth")
    else:
        opt_path = os.path.join(save_folder, "optimizer.pth")
    torch.save(optimizer_state, opt_path)
    
    # Save center loss optimizer
    if center_loss_optimizer_state is not None:
        if epoch is not None:
            center_opt_path = os.path.join(save_folder, "checkpoint", f"center_loss_optimizer_{epoch:02d}.pth")
        else:
            center_opt_path = os.path.join(save_folder, "center_loss_optimizer.pth")
        torch.save(center_loss_optimizer_state, center_opt_path)
        
    # Save stats
    if training_stats is not None:
        if epoch is not None:
            stats_path = os.path.join(save_folder, "checkpoint", f"training_stats_{epoch:02d}.json")
        else:
            stats_path = os.path.join(save_folder, "training_stats.json")
        with open(stats_path, "w") as file:
            file.write(json.dumps(training_stats))