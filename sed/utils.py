import os
import random
import torch
import numpy as np
import json
import re
import time
from pathlib import Path
from torch.utils.data.sampler import Sampler

class SimpleBalanceClassSampler(Sampler):

    def __init__(self, targets, classes_num):

        self.targets = targets
        self.classes_num = classes_num
        self.max_num=100 #hardcode 
        
        self.indexes_per_class = []
        for k in range(self.classes_num):
            self.indexes_per_class.append(
                np.where(self.targets[:, k] == 1)[0])
        
        self.length = self.classes_num * self.max_num

    def __iter__(self):
        
        all_indexs = []
        
        for k in range(self.classes_num):
            if len(self.indexes_per_class[k]) == self.max_num:
                all_indexs.append(self.indexes_per_class[k])
            elif len(self.indexes_per_class[k]) > self.max_num:
                random_choice = np.random.choice(self.indexes_per_class[k], int(self.max_num), replace=True)
                all_indexs.append(np.array(list(random_choice)))
            else:
                gap = self.max_num - len(self.indexes_per_class[k])
                random_choice = np.random.choice(self.indexes_per_class[k], int(gap), replace=True)
                all_indexs.append(np.array(list(random_choice) + list(self.indexes_per_class[k])))
                
        l = np.stack(all_indexs).T
        l = l.reshape(-1)
        random.shuffle(l)
        return iter(l)

    def __len__(self):
        return int(self.length)
        
def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def mixup_data(x, y, alpha=5):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    index = torch.randperm(x.size()[0]).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def smooth_label(y , alpha=0.01):
    y = y * (1 - alpha)
    y[y == 0] = alpha
    return y


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_logger(log_file='train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return loggerutil

class AutoSave:
    def __init__(self, top_k=2, metric="f1", mode="min", root=None, name="ckpt"):
        self.top_k = top_k
        self.logs = []
        self.metric = metric
        self.mode = mode
        self.root = Path(root)
        assert self.root.exists()
        self.name = name

        self.top_models = []
        self.top_metrics = []

    def log(self, model, metrics):
        metric = metrics[self.metric]
        rank = self.rank(metric)

        self.top_metrics.insert(rank+1, metric)
        if len(self.top_metrics) > self.top_k:
            self.top_metrics.pop(0)

        self.logs.append(metrics)
        self.save(model, metric, rank, metrics["epoch"])


    def save(self, model, metric, rank, epoch):
        t = time.strftime("%Y%m%d%H%M%S")
        name = "{}_epoch_{:02d}_{}_{:.04f}_{}".format(self.name, epoch, self.metric, metric, t)
        name = re.sub(r"[^\w_-]", "", name) + ".pth"
        path = self.root.joinpath(name)

        old_model = None
        self.top_models.insert(rank+1, name)
        if len(self.top_models) > self.top_k:
            old_model = self.root.joinpath(self.top_models[0])
            self.top_models.pop(0)      

        torch.save(model.state_dict(), path.as_posix())

        if old_model is not None:
            old_model.unlink()

        self.to_json()


    def rank(self, val):
        r = -1
        for top_val in self.top_metrics:
            if val <= top_val:
                return r
            r += 1

        return r
  
    def to_json(self):
    # t = time.strftime("%Y%m%d%H%M%S")
        name = "{}_logs".format(self.name)
        name = re.sub(r"[^\w_-]", "", name) + ".json"
        path = self.root.joinpath(name)

        with path.open("w") as f:
            json.dump(self.logs, f, indent=2)
