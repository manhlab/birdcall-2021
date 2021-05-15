from sed.engine import *
import pandas as pd
from pathlib import Path
from sed.config import CFG
from sed.utils import *


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