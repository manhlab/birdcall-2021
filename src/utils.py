import argparse
import codecs
import json
import logging
import os
import random
import time

import numpy as np
import torch
import yaml

from contextlib import contextmanager
from typing import Union, Optional
from pathlib import Path
from sklearn.metrics import f1_score, average_precision_score


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def save_json(config: dict, save_path: Union[str, Path]):
    f = codecs.open(str(save_path), mode="w", encoding="utf-8")
    json.dump(config, f, indent=4, cls=MyEncoder, ensure_ascii=False)
    f.close()


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore


@contextmanager
def timer(name: str, logger: Optional[logging.Logger] = None):
    t0 = time.time()
    msg = f"[{name}] start"
    if logger is None:
        print(msg)
    else:
        logger.info(msg)
    yield

    msg = f"[{name}] done in {time.time() - t0:.2f} s"
    if logger is None:
        print(msg)
    else:
        logger.info(msg)


def get_logger(out_file=None):
    logger = logging.getLogger()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger.handlers = []
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    if out_file is not None:
        fh = logging.FileHandler(out_file)
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
    logger.info("logger set up")
    return logger


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Config file path")
    return parser


def get_sed_parser() -> argparse.ArgumentParser:
    parser = get_parser()
    parser.add_argument("--threshold", default=0.7, type=float)
    return parser


def load_config(path: str):
    with open(path) as f:
        config = yaml.safe_load(f)
    return config
def map_score(targ, out):
    targ = targ["clipwise_output"].detach().cpu().numpy()
    clipwise_output = out.detach().cpu().numpy()
    clipwise_output = clipwise_output >= 0.5
    score = average_precision_score(clipwise_output, targ, average=None)
    score = np.nan_to_num(score).mean()
    return score


def f1_score_threashold(targ, out, threshold=0.5):
    targ = targ["clipwise_output"].detach().cpu().numpy()
    clipwise_output = out.detach().cpu().numpy()
    scores = []
    for i in range(len(targ[0])):
        class_i_pred = clipwise_output[:, i] > threshold
        class_i_targ = targ[:, i]

        if class_i_targ.sum() == 0 and class_i_pred.sum() == 0:
            score = 1.0
        else:
            score = f1_score(class_i_pred, class_i_targ.round())
        scores.append(score)

    return np.mean(scores)