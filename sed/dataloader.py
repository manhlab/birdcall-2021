from  torch.utils.data import Dataset, DataLoader
import numpy as np
from .config import CFG
import random
import librosa
from .utils import mono_to_color, random_power
class BirdClefDataset(Dataset):
    def __init__(self,  meta, sr=CFG.sr, is_train=True, num_classes=CFG.num_classes, duration=CFG.duration):
        self.meta = meta.copy().reset_index(drop=True)
        self.sr = sr
        self.is_train = is_train
        self.num_classes = num_classes
        self.duration = duration
        self.audio_length = self.duration*self.sr
        self.n_mels = 128
        self.len_chack = 626
        self.stop_border = 0.3 # Probability of stopping mixing | Вероятность прервать смешивание
        self.level_noise = 0.05 # level noise | Уровень шума
        self.div_coef = 100 # signal amplification during mixing | Усиления сигнала при смешивании

    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        # imagesx = np.load(row.impath)
        ebird_code = row["primary_label"]
        secondary_label = row["secondary_labels"]
        meta_data = np.array(row[["latitude", "longitude"]])
        data = np.array(list(map(int, row["date"].split("-"))))
        meta_data = np.hstack((meta_data, data)).astype(np.float32)
        melspecs = np.load(row['impath'], allow_pickle=True)
        t_pobs = melspecs.item().get('probs')
        try:
            images = melspecs.item().get('images')[np.random.choice(len(t_pobs), size=1, p=t_pobs)[0]]
        except:
            images = melspecs.item().get('images')[np.random.choice(len(t_pobs))]
        t = np.zeros(self.num_classes, dtype=np.float32)  # Label smoothing
        # images = (images+80)/80
        # Add noise | Добавить шум
        # Add white noise | Добавить белый шум 
        t[CFG.target_columns.index(ebird_code)] = 1.0
        if self.is_train:   
             
            if random.random()<0.9:
                    images = images + (np.random.sample((self.n_mels,self.len_chack)).astype(np.float32)+9) * images.mean() * self.level_noise * (np.random.sample() + 0.3)
                
                # Add pink noise | Добавить розовый шум
            if random.random()<0.9:
                    r = random.randint(1,self.n_mels)
                    pink_noise = np.array([np.concatenate((1 - np.arange(r)/r,np.zeros(self.n_mels-r)))]).T
                    images = images + (np.random.sample((self.n_mels,self.len_chack)).astype(np.float32)+9) * 2  * images.mean() * self.level_noise * (np.random.sample() + 0.3)
                
                # Add bandpass noise | Добавить полосовой шум
            if random.random()<0.9:
                    a = random.randint(0, self.n_mels//2)
                    b = random.randint(a+20, self.n_mels)
                    images[a:b,:] = images[a:b,:] + (np.random.sample((b-a,self.len_chack)).astype(np.float32)+9) * 0.05 * images.mean() * self.level_noise  * (np.random.sample() + 0.3)
                
                
                # Lower the upper frequencies | Понизить верхние частоты
            if random.random()<0.5:
                    images = images - images.min()
                    r = random.randint(self.n_mels//2,self.n_mels)
                    x = random.random()/2
                    pink_noise = np.array([np.concatenate((1-np.arange(r)*x/r,np.zeros(self.n_mels-r)-x+1))]).T
                    images = images*pink_noise
                    images = images/(images.max()+0.0000001)

            # if random.random()<0.1:
            #         w = np.random.uniform(0.2, 0.5)
            #         images = (images + w*imagesx[np.random.choice(len(imagesx))])/(1+w)

            if random.random()<0.5:
                    k = np.random.uniform(0.0, 0.7)
                    h = np.random.uniform(k, k+0.3)
                    h = int( h * self.len_chack)
                    k = int( k * self.len_chack)
                    images[:, k:h] = 0

        # Change the contrast | Изменить контрастность
        images = images.astype("float32", copy=False)
        
        images = librosa.power_to_db(images, ref=np.max)
        images = (images+80)/80
        images = random_power(images, power = 2, c= 0.7)
        images = np.nan_to_num(images)
        images = mono_to_color(images)
        for second_label in secondary_label:
            if second_label in CFG.target_columns:
                t[CFG.target_columns.index(second_label)] = 0.3
        return images, meta_data, t


class BirdClefDataset_V2(Dataset):
    def __init__(self,  meta, sr=CFG.sr, is_train=True, num_classes=CFG.num_classes, duration=CFG.duration):
        self.meta = meta.copy().reset_index(drop=True)
        self.sr = sr
        self.is_train = is_train
        self.num_classes = num_classes
        self.duration = duration
        self.audio_length = self.duration*self.sr
        self.n_mels = 128
        self.len_chack = 626
        self.stop_border = 0.8 # Probability of stopping mixing | Вероятность прервать смешивание
        self.level_noise = 0.05 # level noise | Уровень шума
        self.div_coef = 100 # signal amplification during mixing | Усиления сигнала при смешивании

    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, idx):
        idx2 = random.randint(0, len(self.meta)-1) # Second file | Второй файл
        idx3 = random.randint(0, len(self.meta)-1) # Third file | Третий файл
        y = np.zeros(self.num_classes, dtype=np.float32)
        birds, background = [],[]
        images = np.zeros((self.n_mels, self.len_chack)).astype(np.float32)            
        for i,idy in enumerate([idx,idx2,idx3]):
            # Choosing a record with a bird | Выбираем запись с птицей
            row = self.meta.iloc[idy]
            # imagesx = np.load(row.impath)
            ebird_code = row["primary_label"]
            secondary_label = row["secondary_labels"]
            melspecs = np.load(row['impath'], allow_pickle=True)
            t_pobs = melspecs.item().get('probs')
            try:
                melspecs = melspecs.item().get('images')[np.random.choice(len(t_pobs), size=1, p=t_pobs)[0]]
            except:
                melspecs = melspecs.item().get('images')[np.random.choice(len(t_pobs))]

            # Birds in the file | Птицы в файле
            birds.append(CFG.target_columns.index(ebird_code))
            # Birds in the background | Птицы на фоне     
            for second_label in secondary_label:
                if second_label in CFG.target_columns:
                        background.append(CFG.target_columns.index(second_label))
           
            # Change the contrast | Изменить контрастность
            melspecs = random_power(melspecs, power = 3, c= 0.5)
            images = images + melspecs*(random.random() * self.div_coef + 1)            
            if random.random()<self.stop_border:
                break
            if self.is_train:
                break

        images = librosa.power_to_db(images.astype(np.float32), ref=np.max)
        images = (images+80)/80
        if self.is_train:          
            if random.random()<0.9:
                images = images + (np.random.sample((self.n_mels,self.len_chack)).astype(np.float32)+9) * images.mean() * self.level_noise * (np.random.sample() + 0.3)
            
            # Add pink noise | Добавить розовый шум
            if random.random()<0.9:
                r = random.randint(1,self.n_mels)
                pink_noise = np.array([np.concatenate((1 - np.arange(r)/r,np.zeros(self.n_mels-r)))]).T
                images = images + (np.random.sample((self.n_mels,self.len_chack)).astype(np.float32)+9) * 2  * images.mean() * self.level_noise * (np.random.sample() + 0.3)
            
            # Add bandpass noise | Добавить полосовой шум
            if random.random()<0.9:
                a = random.randint(0, self.n_mels//2)
                b = random.randint(a+20, self.n_mels)
                images[a:b,:] = images[a:b,:] + (np.random.sample((b-a,self.len_chack)).astype(np.float32)+9) * 0.05 * images.mean() * self.level_noise  * (np.random.sample() + 0.3)
            
            
            # Lower the upper frequencies | Понизить верхние частоты
            if random.random()<0.5:
                images = images - images.min()
                r = random.randint(self.n_mels//2,self.n_mels)
                x = random.random()/2
                pink_noise = np.array([np.concatenate((1-np.arange(r)*x/r,np.zeros(self.n_mels-r)-x+1))]).T
                images = images*pink_noise
                images = images/images.max()
        
        # Change the contrast | Изменить контрастность
        images = random_power(images, power = 2, c= 0.7)
        images = mono_to_color(images)
        for bird in background:
            if bird < len(y):
                y[bird]=0.3
        for bird in birds:
            #if not bird==264:
            y[bird]=1
        return images, y
