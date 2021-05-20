import numpy as np
import librosa as lb
import librosa.display as lbd
import soundfile as sf
from soundfile import SoundFile
import pandas as pd
from IPython.display import Audio
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
import joblib, json, random, math
import numpy as np
import cv2

SPEC_WIDTH = 256
SR = 32_000
DURATION = 20
FMIN = 300
FMAX = 15_000
NMELS = 128
NFFT = 1024
HOP_LENGTH = 1024
TRAIN_AUDIO_ROOT = Path("/content/bird_background")
TRAIN_AUDIO_IMAGES_SAVE_ROOT = Path("/content/audio_cache")


def get_audio_info(filepath):
    """Get some properties from  an audio file"""
    with SoundFile(filepath) as f:
        sr = f.samplerate
        frames = f.frames
        duration = float(frames) / sr
    return {"frames": frames, "sr": sr, "duration": duration}


def functionfile(t):
    t = t.split("/")
    t = TRAIN_AUDIO_ROOT / f"{t[-2]}/{t[-1]}"
    return t


def crop_or_pad(y, length, is_train=True, start=None):
    if len(y) < length:
        y = np.concatenate([y, np.zeros(length - len(y))])
        n_repeats = length // len(y)
        epsilon = length % len(y)
        y = np.concatenate([y] * n_repeats + [y[:epsilon]])
    elif len(y) > length:
        if not is_train:
            start = start or 0
        else:
            start = start or np.random.randint(len(y) - length)
        y = y[start : start + length]
    return y


class AudioToImage:
    def __init__(self, step=None, res_type="kaiser_fast", resample=True):
        self.audio_length = DURATION * SR
        self.step = step or self.audio_length
        self.res_type = res_type
        self.resample = resample

    def audio_to_image(self, audio):
        image = lb.feature.melspectrogram(
            audio,
            sr=SR,
            n_mels=NMELS,
            fmin=FMIN,
            fmax=FMAX,
            n_fft=NFFT,
            hop_length=HOP_LENGTH,
        )
        return image.astype(np.float16, copy=False)

    def __call__(self, filepath, save=True):
        audio, orig_sr = sf.read(filepath, dtype="float32")
        if self.resample and orig_sr != SR:
            audio = lb.resample(audio, orig_sr, SR, res_type=self.res_type)
        step = int(3 * SR)
        audios = [
            audio[i : i + self.audio_length]
            for i in range(0, max(1, len(audio) - self.audio_length + 1), step)
        ]

        audios[-1] = crop_or_pad(audios[-1], length=self.audio_length)
        images = [self.audio_to_image(audio) for audio in audios]
        images = np.stack(images)

        if save:
            path = TRAIN_AUDIO_IMAGES_SAVE_ROOT / f"{str(filepath).split('/')[-1]}.npy"
            path.parent.mkdir(exist_ok=True, parents=True)
            np.save(str(path), images)
        else:
            return row.filename, images


def get_audios_as_images(filelist):
    pool = joblib.Parallel(4)
    converter = AudioToImage()
    mapper = joblib.delayed(converter)
    tasks = [mapper(row) for row in filelist]
    pool(tqdm(tasks))


if __name__ == "__main__":
    from glob import glob

    listfile = glob(str(TRAIN_AUDIO_ROOT) + "/*")
    get_audios_as_images(listfile)
