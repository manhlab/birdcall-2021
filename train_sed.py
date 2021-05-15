
from src.engine import train
MODEL_NAMES = "resnet50"
MODEL_ROOT = Path("/content/gdrive/MyDrive/Kaggle/kkiller-dataset/RESNET50_SED_MIX3AUDIO")

for model_name in MODEL_NAMES:
    print("\n\n###########################################", model_name.upper())
    try:
        train(model_name, epochs=35, save_root=MODEL_ROOT, suffix=f"_sr{32000}_d{20}_v1_v1")
    except Exception as e:
        raise ValueError() from  e