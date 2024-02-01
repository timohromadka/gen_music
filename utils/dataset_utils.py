import glob
import logging
from io import BytesIO
import os
import pandas as pd
import requests
from tqdm import tqdm

import torchaudio
from torchaudio.transforms import Spectrogram, MelSpectrogram
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
from pytorch_lightning import LightningDataModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('utils/dataset_utils.py')


def get_spectrogram(waveform):
    spectrogram = Spectrogram(n_fft=400)(waveform)
    return spectrogram


def get_mel_spectrogram(waveform, sample_rate):
    mel_spectrogram = MelSpectrogram(sample_rate=sample_rate)(waveform)
    return mel_spectrogram

class AudioDataset(Dataset):
    def __init__(self, df, dataset_type, cache_dir='cache'):
        self.dataset_type = dataset_type
        cache_file = os.path.join(cache_dir, f"{dataset_type}_data.pth")
        
        if os.path.exists(cache_file):
            self.data_paths = torch.load(cache_file)
        else:
            self.data_paths = preprocess_and_cache_dataset(df, dataset_type, cache_dir)
            torch.save(self.data_paths, cache_file)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        file_path = self.data_paths[idx]
        if os.path.exists(file_path):
            waveform = torch.load(file_path)
            return waveform
        else:
            logger.error(f'File not found: {file_path}')
            return None



def preprocess_and_cache_dataset(df, dataset_type, cache_dir='cache'):
    os.makedirs(cache_dir, exist_ok=True)
    
    processed_data = []
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        url = row['SampleURL']
        file_name = f"{idx}.pt"  # PyTorch file
        file_path = os.path.join(cache_dir, file_name)
        
        if not os.path.exists(file_path):
            response = requests.get(url)
            if response.status_code == 200:
                waveform, sample_rate = torchaudio.load(BytesIO(response.content))
                if dataset_type == 'spectrogram':
                    waveform = Spectrogram(n_fft=400)(waveform)
                elif dataset_type == 'mel-spectrogram':
                    waveform = MelSpectrogram(sample_rate=sample_rate)(waveform)
                elif dataset_type == 'waveform':
                    pass  # keep audio in waveform format
                else:
                    raise ValueError("Unknown dataset_type")
                
                torch.save(waveform, file_path)
            else:
                logger.error(f'Failed to download: {url}')
                continue
        processed_data.append(file_path)
        
    return processed_data
        
        
class CustomDataModule(LightningDataModule):
    def __init__(self, train_loader, val_loader, test_loader):
        super().__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


def get_train_val_test_sets(dataset_name, dataset_type, val_split_seed):
    logger.info(f'Fetching train, val, and test sets according to args. dataset_name: {dataset_name}')

    if dataset_name == 'spotify_sleep_dataset':
        df = pd.read_csv('data/SPD_unique_withClusters.csv')
    else:
        raise ValueError("Unknown dataset_name")

    dataset = AudioDataset(df, dataset_type)

    # Split the dataset into train, validation, and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Further split the training dataset into train and validation
    local_generator = torch.Generator().manual_seed(val_split_seed)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=local_generator)

    return train_dataset, val_dataset, test_dataset
    
def get_dataloaders(args):
    train_dataset, val_dataset, test_dataset = get_train_val_test_sets(args.dataset, args.data_type, args.val_split_seed)

    logger.info(f'Fetching dataloaders.')
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    train_unshuffled_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=False) 
    val_loader = DataLoader(val_dataset, batch_size=args.validation_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.validation_batch_size, shuffle=False)

    return train_loader, train_unshuffled_loader, val_loader, test_loader

def preprocess_dataset(dataset):
    return dataset
