
from src.engine import train
import pandas as pd
from pathlib import Path
from src.config import CFG

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