import torch
from sklearn.metrics import label_ranking_average_precision_score
from  torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from tqdm.notebook import tqdm as tqdm_notebook
import numpy as np
from tqdm.notebook import tqdm
from pathlib import Path
import gc
from .utils import *
from .dataloader import *
from .config import CFG
from .model import ResNestSED, EfficientNetSED
from .criterion import *


def one_step( xb,  yb, net,  criterion, optimizer, scheduler=None, mixup_proba=0.5, alpha=5, label_smoothing=True):
    xb, yb = xb.to(CFG.DEVICE),yb.to(CFG.DEVICE)
    optimizer.zero_grad()
    if np.random.rand() < mixup_proba:
            xb, y_a, y_b, _ = mixup_data(xb.cuda(), yb.cuda(), alpha=alpha)
            yb = torch.clamp(y_a + y_b, 0, 1)
    # if label_smoothing:
    #     yb = smooth_label(yb)
    o = net(xb)
    loss = criterion(o, yb)
    
    loss.backward()
    optimizer.step()
    o = o["logit"]
    
    with torch.no_grad():
        l = loss.item()
        o = o.sigmoid()
        yb = (yb > 0.5)*1.0
        lrap = label_ranking_average_precision_score(yb.cpu().numpy(), o.cpu().numpy())
        o = (o > 0.5)*1.0
        prec = (o*yb).sum()/(1e-6 + o.sum())
        rec = (o*yb).sum()/(1e-6 + yb.sum())
        f1 = 2*prec*rec/(1e-6+prec+rec)
    return l, lrap, f1.item(), rec.item(), prec.item()

@torch.no_grad()
def evaluate(net, criterion, val_laoder):
    net.eval()

    os, y = [], []
    val_laoder = tqdm_notebook(val_laoder, leave = False, total=len(val_laoder))

    for icount, (xb, yb) in  enumerate(val_laoder):
        y.append(yb.to(CFG.DEVICE))
        xb = xb.to(CFG.DEVICE)
        o = net(xb)["logit"]
        os.append(o)
    y = torch.cat(y)
    o = torch.cat(os)
    l = nn.BCEWithLogitsLoss()(o, y).item()
    o = o.sigmoid()
    y = (y > 0.5)*1.0
    lrap = label_ranking_average_precision_score(y.cpu().numpy(), o.cpu().numpy())
    o = (o > 0.5)*1.0
    prec = ((o*y).sum()/(1e-6 + o.sum())).item()
    rec = ((o*y).sum()/(1e-6 + y.sum())).item()
    f1 = 2*prec*rec/(1e-6+prec+rec)
    return l, lrap, f1, rec, prec,

def one_epoch(net, criterion, optimizer, scheduler, train_laoder, val_laoder, n=10):
  net.train()
  l, lrap, prec, rec, f1, icount = 0.,0.,0.,0., 0., 0
  train_laoder = tqdm_notebook(train_laoder, leave = False)
  epoch_bar = train_laoder
  cnt = n 
  for (xb, yb) in  epoch_bar:
      # epoch_bar.set_description("----|----|----|----|---->")
      cnt -= 1
      _l, _lrap, _f1, _rec, _prec = one_step(xb, yb, net, criterion, optimizer)
      l += _l
      lrap += _lrap
      f1 += _f1
      rec += _rec
      prec += _prec

      icount += 1
        
      if hasattr(epoch_bar, "set_postfix") and not icount%10:
          epoch_bar.set_postfix(
            loss="{:.6f}".format(l/icount),
            lrap="{:.3f}".format(lrap/icount),
            prec="{:.3f}".format(prec/icount),
            rec="{:.3f}".format(rec/icount),
            f1="{:.3f}".format(f1/icount),
          )
  l /= icount
  lrap /= icount
  f1 /= icount
  rec /= icount
  prec /= icount
  
  l_val, lrap_val, f1_val, rec_val, prec_val = evaluate(net, criterion, val_laoder)

  scheduler.step()  
  return (l, l_val), (lrap, lrap_val), (f1, f1_val), (rec, rec_val), (prec, prec_val)

