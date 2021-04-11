import cv2
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch.utils.data as data

from pathlib import Path


BIRD_CODE = [
        'acafly', 'acowoo', 'aldfly', 'ameavo', 'amecro',
        'amegfi', 'amekes', 'amepip', 'amered', 'amerob',
        'amewig', 'amtspa', 'andsol1', 'annhum', 'astfly',
        'azaspi1', 'babwar', 'baleag', 'balori', 'banana',
        'banswa', 'banwre1', 'barant1', 'barswa', 'batpig1',
        'bawswa1', 'bawwar', 'baywre1', 'bbwduc', 'bcnher',
        'belkin1', 'belvir', 'bewwre', 'bkbmag1', 'bkbplo',
        'bkbwar', 'bkcchi', 'bkhgro', 'bkmtou1', 'bknsti', 'blbgra1',
        'blbthr1', 'blcjay1', 'blctan1', 'blhpar1', 'blkpho',
        'blsspa1', 'blugrb1', 'blujay', 'bncfly', 'bnhcow', 'bobfly1',
        'bongul', 'botgra', 'brbmot1', 'brbsol1', 'brcvir1', 'brebla',
        'brncre', 'brnjay', 'brnthr', 'brratt1', 'brwhaw', 'brwpar1',
        'btbwar', 'btnwar', 'btywar', 'bucmot2', 'buggna', 'bugtan',
        'buhvir', 'bulori', 'burwar1', 'bushti', 'butsal1', 'buwtea',
        'cacgoo1', 'cacwre', 'calqua', 'caltow', 'cangoo', 'canwar',
        'carchi', 'carwre', 'casfin', 'caskin', 'caster1', 'casvir',
        'categr', 'ccbfin', 'cedwax', 'chbant1', 'chbchi', 'chbwre1',
        'chcant2', 'chispa', 'chswar', 'cinfly2', 'clanut', 'clcrob',
        'cliswa', 'cobtan1', 'cocwoo1', 'cogdov', 'colcha1', 'coltro1',
        'comgol', 'comgra', 'comloo', 'commer', 'compau', 'compot1',
        'comrav', 'comyel', 'coohaw', 'cotfly1', 'cowscj1', 'cregua1',
        'creoro1', 'crfpar', 'cubthr', 'daejun', 'dowwoo', 'ducfly', 'dusfly',
        'easblu', 'easkin', 'easmea', 'easpho', 'eastow', 'eawpew', 'eletro',
        'eucdov', 'eursta', 'fepowl', 'fiespa', 'flrtan1', 'foxspa', 'gadwal',
        'gamqua', 'gartro1', 'gbbgul', 'gbwwre1', 'gcrwar', 'gilwoo',
        'gnttow', 'gnwtea', 'gocfly1', 'gockin', 'gocspa', 'goftyr1',
        'gohque1', 'goowoo1', 'grasal1', 'grbani', 'grbher3', 'grcfly',
        'greegr', 'grekis', 'grepew', 'grethr1', 'gretin1', 'greyel',
        'grhcha1', 'grhowl', 'grnher', 'grnjay', 'grtgra', 'grycat',
        'gryhaw2', 'gwfgoo', 'haiwoo', 'heptan', 'hergul', 'herthr',
        'herwar', 'higmot1', 'hofwoo1', 'houfin', 'houspa', 'houwre',
        'hutvir', 'incdov', 'indbun', 'kebtou1', 'killde', 'labwoo', 'larspa',
        'laufal1', 'laugul', 'lazbun', 'leafly', 'leasan', 'lesgol', 'lesgre1',
        'lesvio1', 'linspa', 'linwoo1', 'littin1', 'lobdow', 'lobgna5', 'logshr',
        'lotduc', 'lotman1', 'lucwar', 'macwar', 'magwar', 'mallar3', 'marwre',
        'mastro1', 'meapar', 'melbla1', 'monoro1', 'mouchi', 'moudov', 'mouela1',
        'mouqua', 'mouwar', 'mutswa', 'naswar', 'norcar', 'norfli', 'normoc', 'norpar',
        'norsho', 'norwat', 'nrwswa', 'nutwoo', 'oaktit', 'obnthr1', 'ocbfly1',
        'oliwoo1', 'olsfly', 'orbeup1', 'orbspa1', 'orcpar', 'orcwar', 'orfpar',
        'osprey', 'ovenbi1', 'pabspi1', 'paltan1', 'palwar', 'pasfly', 'pavpig2',
        'phivir', 'pibgre', 'pilwoo', 'pinsis', 'pirfly1', 'plawre1', 'plaxen1',
        'plsvir', 'plupig2', 'prowar', 'purfin', 'purgal2', 'putfru1', 'pygnut',
        'rawwre1', 'rcatan1', 'rebnut', 'rebsap', 'rebwoo', 'redcro', 'reevir1',
        'rehbar1', 'relpar', 'reshaw', 'rethaw', 'rewbla', 'ribgul', 'rinkin1',
        'roahaw', 'robgro', 'rocpig', 'rotbec', 'royter1', 'rthhum', 'rtlhum',
        'ruboro1', 'rubpep1', 'rubrob', 'rubwre1', 'ruckin', 'rucspa1', 'rucwar',
        'rucwar1', 'rudpig', 'rudtur', 'rufhum', 'rugdov', 'rumfly1', 'runwre1',
        'rutjac1', 'saffin', 'sancra', 'sander', 'savspa', 'saypho', 'scamac1',
        'scatan', 'scbwre1', 'scptyr1', 'scrtan1', 'semplo', 'shicow', 'sibtan2',
        'sinwre1', 'sltred', 'smbani', 'snogoo', 'sobtyr1', 'socfly1', 'solsan',
        'sonspa', 'soulap1', 'sposan', 'spotow', 'spvear1', 'squcuc1', 'stbori',
        'stejay', 'sthant1', 'sthwoo1', 'strcuc1', 'strfly1', 'strsal1', 'stvhum2',
        'subfly', 'sumtan', 'swaspa', 'swathr', 'tenwar', 'thbeup1', 'thbkin',
        'thswar1', 'towsol', 'treswa', 'trogna1', 'trokin', 'tromoc', 'tropar',
        'tropew1', 'tuftit', 'tunswa', 'veery', 'verdin', 'vigswa', 'warvir',
        'wbwwre1', 'webwoo1', 'wegspa1', 'wesant1', 'wesblu', 'weskin', 'wesmea',
        'westan', 'wewpew', 'whbman1', 'whbnut', 'whcpar', 'whcsee1', 'whcspa',
        'whevir', 'whfpar1', 'whimbr', 'whiwre1', 'whtdov', 'whtspa', 'whwbec1',
        'whwdov', 'wilfly', 'willet1', 'wilsni1', 'wiltur', 'wlswar', 'wooduc',
        'woothr', 'wrenti', 'y00475', 'yebcha', 'yebela1', 'yebfly', 'yebori1',
        'yebsap', 'yebsee1', 'yefgra1', 'yegvir', 'yehbla', 'yehcar1', 'yelgro',
        'yelwar', 'yeofly1', 'yerwar', 'yeteup1', 'yetvir']

PERIOD = 5

class PANNsMultiLabelDataset(data.Dataset):
    def __init__(self, df: pd.DataFrame, datadir: Path, transforms=None, period=30):
        self.df = df
        self.datadir = datadir
        self.transforms = transforms
        self.period = period

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]
        wav_name = sample["resampled_filename"]
        ebird_code = sample["ebird_code"]
        secondary_labels = eval(sample["secondary_labels"])
        y, sr = sf.read(self.datadir / ebird_code / wav_name)

        len_y = len(y)
        effective_length = sr * self.period
        if len_y < effective_length:
            new_y = np.zeros(effective_length, dtype=y.dtype)
            start = np.random.randint(effective_length - len_y)
            new_y[start:start + len_y] = y
            y = new_y.astype(np.float32)
        elif len_y > effective_length:
            start = np.random.randint(len_y - effective_length)
            y = y[start:start + effective_length].astype(np.float32)
        else:
            y = y.astype(np.float32)

        if self.transforms:
            y = self.transforms(y)

        labels = np.zeros(len(BIRD_CODE), dtype=int)
        labels[BIRD_CODE[ebird_code]] = 1
        for second_label in secondary_labels:
            if NAME2CODE.get(second_label) is not None:
                second_code = NAME2CODE[second_label]
                labels[BIRD_CODE[second_code]] = 1

        return {
            "waveform": y,
            "targets": labels
        }


class PANNsSedDataset(data.Dataset):
    def __init__(self, df: pd.DataFrame, datadir: Path, transforms=None,
                 denoised_audio_dir=None):
        self.df = df
        self.datadir = datadir
        self.transforms = transforms
        self.denoised_audio_dir = denoised_audio_dir
        if denoised_audio_dir is not None:
            self.use_denoised = True
        else:
            self.use_denoised = False

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]
        wav_name = sample["resampled_filename"]
        ebird_code = sample["ebird_code"]
        secondary_labels = eval(sample["secondary_labels"])

        if self.use_denoised:
            path = self.denoised_audio_dir / ebird_code / wav_name
            if path.exists():
                y, sr = sf.read(path)
            else:
                y, sr = sf.read(self.datadir / ebird_code / wav_name)
        else:
            y, sr = sf.read(self.datadir / ebird_code / wav_name)

        duration = len(y) / sr
        if self.transforms:
            y = self.transforms(y)

        audios = []
        len_y = len(y)
        start = 0
        end = sr * 5
        while len_y > start:
            y_batch = y[start:end].astype(np.float32)
            if len(y_batch) != (sr * 5):
                y_batch_large = np.zeros(sr * 5, dtype=y_batch.dtype)
                y_batch_large[:len(y_batch)] = y_batch
                audios.append(y_batch_large)
                break
            start = end
            end = end + sr * 5

            audios.append(y_batch)
        audios = np.asarray(audios).astype(np.float32)

        labels = np.zeros(len(BIRD_CODE), dtype=int)
        labels[BIRD_CODE[ebird_code]] = 1
        for secondary_label in secondary_labels:
            code = NAME2CODE.get(secondary_label)
            if code is None:
                continue
            else:
                labels[
                    BIRD_CODE[code]
                ] = 1

        return {
            "waveform": audios,
            "targets": labels,
            "ebird_code": ebird_code,
            "wav_name": wav_name,
            "duration": duration
        }


class NormalizedChannelsSedDataset(data.Dataset):
    def __init__(self, df: pd.DataFrame, datadir: Path, transforms=None,
                 denoised_audio_dir=None, melspectrogram_parameters={},
                 pcen_parameters={},
                 period=30):
        self.df = df
        self.datadir = datadir
        self.transforms = transforms
        self.denoised_audio_dir = denoised_audio_dir
        if denoised_audio_dir is not None:
            self.use_denoised = True
        else:
            self.use_denoised = False
        self.melspectrogram_parameters = melspectrogram_parameters
        self.pcen_parameters = pcen_parameters
        self.period = period

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]
        wav_name = sample["resampled_filename"]
        ebird_code = sample["ebird_code"]
        secondary_labels = eval(sample["secondary_labels"])

        if self.use_denoised:
            path = self.denoised_audio_dir / ebird_code / wav_name
            if path.exists():
                y, sr = sf.read(path)
            else:
                y, sr = sf.read(self.datadir / ebird_code / wav_name)
        else:
            y, sr = sf.read(self.datadir / ebird_code / wav_name)

        duration = len(y) / sr
        if self.transforms:
            y = self.transforms(y)

        images = []
        len_y = len(y)
        if len(y) > 0:
            max_vol = np.abs(y).max()
            if max_vol > 0:
                y = np.asfortranarray(y * 1 / max_vol)
        start = 0
        end = sr * self.period
        while len_y > start:
            y_batch = y[start:end].astype(np.float32)
            if len(y_batch) != (sr * self.period):
                y_batch_large = np.zeros(sr * self.period, dtype=y_batch.dtype)
                y_batch_large[:len(y_batch)] = y_batch

                melspec = librosa.feature.melspectrogram(
                    y_batch_large, sr=sr, **self.melspectrogram_parameters)
                pcen = librosa.pcen(melspec, sr=sr, **self.pcen_parameters)
                clean_mel = librosa.power_to_db(melspec ** 1.5)
                melspec = librosa.power_to_db(melspec)

                norm_melspec = normalize_melspec(melspec)
                norm_pcen = normalize_melspec(pcen)
                norm_clean_mel = normalize_melspec(clean_mel)
                image = np.stack([norm_melspec, norm_pcen, norm_clean_mel], axis=-1)

                height, width, _ = image.shape
                image = cv2.resize(image, (int(width * 224 / height), 224))
                image = np.moveaxis(image, 2, 0)
                image = (image / 255.0).astype(np.float32)

                images.append(image)
                break
            start = end
            end = end + sr * self.period

            melspec = librosa.feature.melspectrogram(
                y_batch, sr=sr, **self.melspectrogram_parameters)
            pcen = librosa.pcen(melspec, sr=sr, **self.pcen_parameters)
            clean_mel = librosa.power_to_db(melspec ** 1.5)
            melspec = librosa.power_to_db(melspec)

            norm_melspec = normalize_melspec(melspec)
            norm_pcen = normalize_melspec(pcen)
            norm_clean_mel = normalize_melspec(clean_mel)
            image = np.stack([norm_melspec, norm_pcen, norm_clean_mel], axis=-1)
            height, width, _ = image.shape
            image = cv2.resize(image, (int(width * 224 / height), 224))
            image = np.moveaxis(image, 2, 0)
            image = (image / 255.0).astype(np.float32)

            images.append(image)
        images = np.asarray(images).astype(np.float32)

        labels = np.zeros(len(BIRD_CODE), dtype=int)
        labels[BIRD_CODE[ebird_code]] = 1
        for secondary_label in secondary_labels:
            code = NAME2CODE.get(secondary_label)
            if code is None:
                continue
            else:
                labels[
                    BIRD_CODE[code]
                ] = 1

        return {
            "image": images,
            "targets": labels,
            "ebird_code": ebird_code,
            "wav_name": wav_name,
            "duration": duration,
            "period": self.period
        }


class ChannelsSedDataset(data.Dataset):
    def __init__(self, df: pd.DataFrame, datadir: Path, transforms=None,
                 denoised_audio_dir=None, melspectrogram_parameters={},
                 pcen_parameters={},
                 period=30):
        self.df = df
        self.datadir = datadir
        self.transforms = transforms
        self.denoised_audio_dir = denoised_audio_dir
        if denoised_audio_dir is not None:
            self.use_denoised = True
        else:
            self.use_denoised = False
        self.melspectrogram_parameters = melspectrogram_parameters
        self.pcen_parameters = pcen_parameters
        self.period = period

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]
        wav_name = sample["resampled_filename"]
        ebird_code = sample["ebird_code"]
        secondary_labels = eval(sample["secondary_labels"])

        if self.use_denoised:
            path = self.denoised_audio_dir / ebird_code / wav_name
            if path.exists():
                y, sr = sf.read(path)
            else:
                y, sr = sf.read(self.datadir / ebird_code / wav_name)
        else:
            y, sr = sf.read(self.datadir / ebird_code / wav_name)

        duration = len(y) / sr
        if self.transforms:
            y = self.transforms(y)

        images = []
        len_y = len(y)
        start = 0
        end = sr * self.period
        while len_y > start:
            y_batch = y[start:end].astype(np.float32)
            if len(y_batch) != (sr * self.period):
                y_batch_large = np.zeros(sr * self.period, dtype=y_batch.dtype)
                y_batch_large[:len(y_batch)] = y_batch

                melspec = librosa.feature.melspectrogram(
                    y_batch_large, sr=sr, **self.melspectrogram_parameters)
                pcen = librosa.pcen(melspec, sr=sr, **self.pcen_parameters)
                clean_mel = librosa.power_to_db(melspec ** 1.5)
                melspec = librosa.power_to_db(melspec)

                norm_melspec = normalize_melspec(melspec)
                norm_pcen = normalize_melspec(pcen)
                norm_clean_mel = normalize_melspec(clean_mel)
                image = np.stack([norm_melspec, norm_pcen, norm_clean_mel], axis=-1)

                height, width, _ = image.shape
                image = cv2.resize(image, (int(width * 224 / height), 224))
                image = np.moveaxis(image, 2, 0)
                image = (image / 255.0).astype(np.float32)

                images.append(image)
                break
            start = end
            end = end + sr * self.period

            melspec = librosa.feature.melspectrogram(
                y_batch, sr=sr, **self.melspectrogram_parameters)
            pcen = librosa.pcen(melspec, sr=sr, **self.pcen_parameters)
            clean_mel = librosa.power_to_db(melspec ** 1.5)
            melspec = librosa.power_to_db(melspec)

            norm_melspec = normalize_melspec(melspec)
            norm_pcen = normalize_melspec(pcen)
            norm_clean_mel = normalize_melspec(clean_mel)
            image = np.stack([norm_melspec, norm_pcen, norm_clean_mel], axis=-1)
            height, width, _ = image.shape
            image = cv2.resize(image, (int(width * 224 / height), 224))
            image = np.moveaxis(image, 2, 0)
            image = (image / 255.0).astype(np.float32)

            images.append(image)
        images = np.asarray(images).astype(np.float32)

        labels = np.zeros(len(BIRD_CODE), dtype=int)
        labels[BIRD_CODE[ebird_code]] = 1
        for secondary_label in secondary_labels:
            code = NAME2CODE.get(secondary_label)
            if code is None:
                continue
            else:
                labels[
                    BIRD_CODE[code]
                ] = 1

        return {
            "image": images,
            "targets": labels,
            "ebird_code": ebird_code,
            "wav_name": wav_name,
            "duration": duration,
            "period": self.period
        }


class MultiChannelDataset(data.Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 datadir: Path,
                 img_size=224,
                 waveform_transforms=None,
                 spectrogram_transforms=None,
                 melspectrogram_parameters={},
                 pcen_parameters={},
                 period=30):
        self.df = df
        self.datadir = datadir
        self.img_size = img_size
        self.waveform_transforms = waveform_transforms
        self.spectrogram_transforms = spectrogram_transforms
        self.melspectrogram_parameters = melspectrogram_parameters
        self.pcen_parameters = pcen_parameters
        self.period = period

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]
        wav_name = sample["resampled_filename"]
        ebird_code = sample["ebird_code"]
        secondary_labels = eval(sample["secondary_labels"])

        y, sr = sf.read(self.datadir / ebird_code / wav_name)

        len_y = len(y)
        effective_length = sr * self.period
        if len_y < effective_length:
            new_y = np.zeros(effective_length, dtype=y.dtype)
            start = np.random.randint(effective_length - len_y)
            new_y[start:start + len_y] = y
            y = new_y.astype(np.float32)
        elif len_y > effective_length:
            start = np.random.randint(len_y - effective_length)
            y = y[start:start + effective_length].astype(np.float32)
        else:
            y = y.astype(np.float32)

        if self.waveform_transforms:
            y = self.waveform_transforms(y)

        melspec = librosa.feature.melspectrogram(y, sr=sr, **self.melspectrogram_parameters)
        pcen = librosa.pcen(melspec, sr=sr, **self.pcen_parameters)
        clean_mel = librosa.power_to_db(melspec ** 1.5)
        melspec = librosa.power_to_db(melspec)

        if self.spectrogram_transforms:
            melspec = self.spectrogram_transforms(image=melspec)["image"]
            pcen = self.spectrogram_transforms(image=pcen)["image"]
            clean_mel = self.spectrogram_transforms(image=clean_mel)["image"]
        else:
            pass

        norm_melspec = normalize_melspec(melspec)
        norm_pcen = normalize_melspec(pcen)
        norm_clean_mel = normalize_melspec(clean_mel)
        image = np.stack([norm_melspec, norm_pcen, norm_clean_mel], axis=-1)

        height, width, _ = image.shape
        image = cv2.resize(image, (int(width * self.img_size / height), self.img_size))
        image = np.moveaxis(image, 2, 0)
        image = (image / 255.0).astype(np.float32)

        labels = np.zeros(len(BIRD_CODE), dtype=int)
        labels[BIRD_CODE[ebird_code]] = 1
        for second_label in secondary_labels:
            if NAME2CODE.get(second_label) is not None:
                second_code = NAME2CODE[second_label]
                labels[BIRD_CODE[second_code]] = 1

        return {
            "image": image,
            "targets": labels
        }


class LabelCorrectionDataset(data.Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 datadir: Path,
                 soft_label_dir: Path,
                 img_size=224,
                 waveform_transforms=None,
                 spectrogram_transforms=None,
                 melspectrogram_parameters={},
                 pcen_parameters={},
                 period=30,
                 n_segments=103,
                 threshold=0.5):
        self.df = df
        self.datadir = datadir
        self.soft_label_dir = soft_label_dir
        self.img_size = img_size
        self.waveform_transforms = waveform_transforms
        self.spectrogram_transforms = spectrogram_transforms
        self.melspectrogram_parameters = melspectrogram_parameters
        self.pcen_parameters = pcen_parameters
        self.period = period
        self.n_segments = n_segments
        self.threshold = threshold

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]
        wav_name = sample["resampled_filename"]
        ebird_code = sample["ebird_code"]
        secondary_labels = eval(sample["secondary_labels"])

        y, sr = sf.read(self.datadir / ebird_code / wav_name)
        soft_label = np.load(self.soft_label_dir / (wav_name + ".npy"))

        sec_per_segment = self.period / self.n_segments
        sec_per_timestep = 1 / sr
        step_per_segment = int(sec_per_segment / sec_per_timestep)

        len_y = len(y)
        effective_length = sr * self.period
        if len_y < effective_length:
            new_y = np.zeros(effective_length, dtype=y.dtype)
            max_offset = effective_length - len_y

            offset_id = np.random.randint(0, (max_offset // step_per_segment) + 1)
            start = offset_id * step_per_segment

            new_y[start:start + len_y] = y
            y = new_y.astype(np.float32)
        elif len_y > effective_length:
            max_offset = len_y - effective_length

            offset_id = np.random.randint(0, (max_offset // step_per_segment) + 1)
            start = offset_id * step_per_segment
            y = y[start:start + effective_length].astype(np.float32)
        else:
            y = y.astype(np.float32)

        if self.waveform_transforms:
            y = self.waveform_transforms(y)

        melspec = librosa.feature.melspectrogram(y, sr=sr, **self.melspectrogram_parameters)
        pcen = librosa.pcen(melspec, sr=sr, **self.pcen_parameters)
        clean_mel = librosa.power_to_db(melspec ** 1.5)
        melspec = librosa.power_to_db(melspec)

        if self.spectrogram_transforms:
            melspec = self.spectrogram_transforms(image=melspec)["image"]
            pcen = self.spectrogram_transforms(image=pcen)["image"]
            clean_mel = self.spectrogram_transforms(image=clean_mel)["image"]
        else:
            pass

        norm_melspec = normalize_melspec(melspec)
        norm_pcen = normalize_melspec(pcen)
        norm_clean_mel = normalize_melspec(clean_mel)
        image = np.stack([norm_melspec, norm_pcen, norm_clean_mel], axis=-1)

        height, width, _ = image.shape
        image = cv2.resize(image, (int(width * self.img_size / height), self.img_size))
        image = np.moveaxis(image, 2, 0)
        image = (image / 255.0).astype(np.float32)

        labels = np.zeros([self.n_segments, len(BIRD_CODE)], dtype=np.float32)

        if len_y < effective_length:
            if len(soft_label) + offset_id >= len(labels):
                n_seg = len(labels[offset_id:, :])
                labels[offset_id:, :] = soft_label[:n_seg]
            else:
                labels[offset_id:offset_id + len(soft_label), :] = soft_label
        elif len_y > effective_length:
            use_labels = soft_label[offset_id:offset_id + len(labels)]
            if len(use_labels) < len(labels):
                labels[:len(use_labels)] = use_labels
            else:
                labels = use_labels
        else:
            if len(labels) >= len(soft_label):
                labels[:len(soft_label)] = soft_label
            else:
                labels = soft_label[:len(labels)]

        labels = labels.astype(np.float32)

        weak_labels = np.zeros(len(BIRD_CODE), dtype=int)
        weak_labels[BIRD_CODE[ebird_code]] = 1
        for second_label in secondary_labels:
            if NAME2CODE.get(second_label) is not None:
                second_code = NAME2CODE[second_label]
                weak_labels[BIRD_CODE[second_code]] = 1
        weak_labels_soft = labels.max(axis=0)
        weak_labels_bin = (weak_labels_soft >= self.threshold).astype(int)

        weak_labels = np.logical_and(weak_labels, weak_labels_bin).astype(int)
        weak_sum_target = labels.sum(axis=0)

        return {
            "image": image,
            "targets": weak_labels,
            "weak_targets": weak_labels,
            "weak_sum_targets": weak_sum_target
        }


def normalize_melspec(X: np.ndarray):
    eps = 1e-6
    mean = X.mean()
    X = X - mean
    std = X.std()
    Xstd = X / (std + eps)
    norm_min, norm_max = Xstd.min(), Xstd.max()
    if (norm_max - norm_min) > eps:
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V


def mono_to_color(X: np.ndarray,
                  mean=None,
                  std=None,
                  norm_max=None,
                  norm_min=None,
                  eps=1e-6):
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V