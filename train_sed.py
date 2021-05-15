
from sed.engine import *
import pandas as pd
from pathlib import Path
from sed.config import CFG
from sed.utils import *

def one_fold(model_name, fold, train_set, val_set, epochs=20, save=True, save_root=None, balance_sample=True):

  save_root = Path(save_root) or CFG.MODEL_ROOT
  saver = AutoSave(root=save_root, name=f"birdclef_{model_name}_fold{fold}", metric="f1_val")
  config_model =   {"base_model_name": "efficientnet-b1",
    "pretrained": True,
    "num_classes": 397}

  # net =  EfficientNetSED("efficientnet-b1", True, 397).to(DEVICE)
  net =  ResNestSED("resnest50", True, 397).to(CFG.DEVICE)
  
  #resnext_meta().to(DEVICE)
  criterion = ImprovedPANNsLoss(weights=[1.0 , 0.5])
  optimizer = optim.AdamW(net.parameters(), lr=CFG.lr)
  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,  T_max=50)
  train_data = BirdClefDataset_V2( meta=train_set, sr=CFG.sr, duration=CFG.duration, is_train=True)
  if balance_sample:
    all_targets = []
    for i in range(len(train_set)):
      ebird_code = train_set.iloc[i]["primary_label"]
      labels = np.zeros(397, dtype="f")
      labels[CFG.target_columns.index(ebird_code)] = 1
      all_targets.append(labels)
    all_targets = np.array(all_targets)
    train_laoder = DataLoader(train_data, batch_size=CFG.batch_size, num_workers=CFG.num_workers, sampler=SimpleBalanceClassSampler(all_targets, 397), pin_memory=True)
  else:
    train_laoder = DataLoader(train_data, batch_size=CFG.batch_size, num_workers=CFG.num_workers, shuffle=True, pin_memory=True)
  val_data = BirdClefDataset_V2( meta=val_set,  sr=CFG.sr, duration=CFG.duration, is_train=False)
  val_laoder = DataLoader(val_data, batch_size=CFG.batch_size, num_workers=CFG.num_workers, shuffle=False, pin_memory=True)
  epochs_bar = tqdm(list(range(epochs)), leave=False)
  for epoch  in epochs_bar:
    epochs_bar.set_description(f"--> [EPOCH {epoch:02d}]")
    net.train()
    (l, l_val), (lrap, lrap_val), (f1, f1_val), (rec, rec_val), (prec, prec_val) = one_epoch(
        net=net,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_laoder=train_laoder,
        val_laoder=val_laoder,
      )
    epochs_bar.set_postfix(
    loss="({:.6f}, {:.6f})".format(l, l_val),
    prec="({:.3f}, {:.3f})".format(prec, prec_val),
    rec="({:.3f}, {:.3f})".format(rec, rec_val),
    f1="({:.3f}, {:.3f})".format(f1, f1_val),
    lrap="({:.3f}, {:.3f})".format(lrap, lrap_val),
    )
    print(
        "[{epoch:02d}] loss: {loss} lrap: {lrap} f1: {f1} rec: {rec} prec: {prec}".format(
            epoch=epoch,
            loss="({:.6f}, {:.6f})".format(l, l_val),
            prec="({:.3f}, {:.3f})".format(prec, prec_val),
            rec="({:.3f}, {:.3f})".format(rec, rec_val),
            f1="({:.3f}, {:.3f})".format(f1, f1_val),
            lrap="({:.3f}, {:.3f})".format(lrap, lrap_val),
        )
    )
    if save:
      metrics = {
          "loss": l, "lrap": lrap, "f1": f1, "rec": rec, "prec": prec,
          "loss_val": l_val, "lrap_val": lrap_val, "f1_val": f1_val, "rec_val": rec_val, "prec_val": prec_val,
          "epoch": epoch,
      }
      saver.log(net, metrics)
  torch.save(net.state_dict(), save_root/f"last_epochs_fold{fold}.pth")
  
def train(df,model_name, epochs=20, save=True, n_splits=5, seed=177, save_root=None, suffix=""):
    gc.collect()
    torch.cuda.empty_cache()
    # environment
    set_seed(CFG.seed)
    device = get_device()
    # validation
    # data
    save_root.mkdir(exist_ok=True, parents=True)
    for i in range(5):
        if i not in CFG.folds:
            continue
        save_root = CFG.MODEL_ROOT/f"fold-{i}"
        save_root.mkdir(exist_ok=True, parents=True)

        print("=" * 120)
        print(f"Fold {i} Training")
        print("=" * 120)
        trn_df = df[df['fold']!=i].reset_index(drop=True)
        val_df = df[df['fold']==i].reset_index(drop=True)
        one_fold(model_name, fold=i, train_set=trn_df , val_set=val_df , epochs=CFG.epochs, save=save, save_root=save_root)
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":  
    MEL_PATHS = "/content/gdrive/MyDrive/Kaggle/kkiller-dataset/rich_train_metadata.csv"
    audio_path = Path("/content/audio_images") 
    MODEL_NAMES = "resnet50"
    MODEL_ROOT = Path("/content/gdrive/MyDrive/Kaggle/kkiller-dataset/RESNET50_SED_MIX3AUDIO")
    df = pd.read_csv(MEL_PATHS)
    df["impath"] = df.apply(lambda row: audio_path/"{}/{}.npy".format(row.primary_label, row.filename), axis=1) 
    for model_name in MODEL_NAMES:
        print("\n\n###########################################", model_name.upper())
        try:
            train(df, model_name, epochs=35, save_root=MODEL_ROOT, suffix=f"_sr{32000}_d{20}_v1_v1")
        except Exception as e:
            raise ValueError() from  e