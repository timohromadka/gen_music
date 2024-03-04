import glob
import logging
from io import BytesIO
import numpy as np
import os
import pandas as pd
import requests
from tqdm import tqdm

import matplotlib.pyplot as plt
import torchaudio
from torchaudio.transforms import Spectrogram, MelSpectrogram, Resample
# downmixmono (conersion from st to mo) has been deprecated, must be done manually
# https://discuss.pytorch.org/t/module-torchaudio-transforms-has-no-attribute-downmixmono/60781

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset, TensorDataset
from pytorch_lightning import LightningDataModule

from utils.musiccaps_utils import preprocess_and_cache_musiccaps_dataset
from utils.spotifysleepdataset_utils import preprocess_and_cache_spotifysleep_dataset
from utils.constants import SAMPLE_LENGTH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('utils/dataset_utils.py')

def get_spectrogram(waveform):
    spectrogram = Spectrogram(n_fft=400)(waveform)
    return spectrogram


def get_mel_spectrogram(waveform, sample_rate, n_mels):
    mel_spectrogram = MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)(waveform)
    return mel_spectrogram

def calculate_waveform_length(sample_length, sample_rate):
    return sample_length * sample_rate

def get_trimmed_waveform(waveform, sample_rate, sample_length, trim_area='random'):
    total_samples = waveform.shape[1]
    num_samples_to_trim = sample_length * sample_rate

    if total_samples <= num_samples_to_trim:
        return waveform

    if trim_area == 'start':
        trimmed_waveform = waveform[:, :num_samples_to_trim]
    elif trim_area == 'random':
        max_start_point = total_samples - num_samples_to_trim
        start_point = np.random.randint(0, max_start_point)
        trimmed_waveform = waveform[:, start_point:start_point + num_samples_to_trim]
    elif trim_area == 'end':
        trimmed_waveform = waveform[:, -num_samples_to_trim:]
    else:
        raise ValueError("trim_area must be 'start', 'random', or 'end'")

    return trimmed_waveform
    



class AudioDataset(Dataset):
    def __init__(self, args):
        self.dataset_type = args.dataset_type
        self.cache_dir = os.path.join(args.cache_dir, args.dataset, self.dataset_type)
        self.sample_rate = args.sample_rate
        self.trim_area = args.trim_area
        self.sample_length = SAMPLE_LENGTH[args.dataset] if not args.sample_length else args.sample_length
        
        os.makedirs(self.cache_dir, exist_ok=True)

        self.data_paths = self.preprocess_and_cache_dataset(args, self.cache_dir)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        file_path = self.data_paths[idx]
        if os.path.exists(file_path):
            waveform, sample_rate = torch.load(file_path)
            # Trim the waveform as necessary using the get_trimmed_waveform function
            waveform = get_trimmed_waveform(waveform, sample_rate, self.sample_length, trim_area=self.trim_area)
            return waveform, sample_rate
        else:
            logger.error(f'File not found: {file_path}')
            return None, None
        
    def preprocess_and_cache_dataset(self, args, cache_dir):
        if args.dataset == 'spotify_sleep_dataset':
            return preprocess_and_cache_spotifysleep_dataset(args, cache_dir)
        elif args.dataset == 'musiccaps':
            return preprocess_and_cache_musiccaps_dataset(args, cache_dir)
        else:
            raise ValueError(f'Unknown dataset specified: {args.dataset}.')
        
        
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


def get_train_val_test_sets(args):
    logger.info(f'Initializing dataset: {args.dataset}')
    
    # ======================
    # Get dataset object
    # ======================
    if args.dataset == 'random':
        # Generate random data
        batch_size = args.train_batch_size  # Assuming batch_size is defined in args
        num_channels = args.num_channels
        length = 2**17  # Fixed length as specified
        # Randomly generate audio data
        random_audio_data = torch.randn(batch_size, num_channels, length)
        # Create a TensorDataset with random audio data and dummy sample rates (if necessary)
        sample_rate = args.sample_rate
        dataset = TensorDataset(random_audio_data, torch.full((batch_size,), sample_rate))
    else:
        dataset = AudioDataset(args)

    logger.info(f'Splitting into train, validation, and test.')
    # Set a fixed seed for reproducible initial split
    generator = torch.Generator().manual_seed(args.val_split_seed)

    # 80% train, 10% val, 10% test
    train_size = int(0.8 * len(dataset))
    temp_size = len(dataset) - train_size
    train_dataset, temp_dataset = random_split(dataset, [train_size, temp_size], generator=generator)
    
    val_size = test_size = int(temp_size / 2)
    val_dataset, test_dataset = random_split(temp_dataset, [val_size, test_size], generator=generator)

    logger.info(f'Finished splitting. Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}')

    return train_dataset, val_dataset, test_dataset
    
def get_dataloaders(args):
    train_dataset, val_dataset, test_dataset = get_train_val_test_sets(args)

    logger.info(f'Fetching dataloaders.')
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    train_unshuffled_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=False) 
    val_loader = DataLoader(val_dataset, batch_size=args.validation_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.validation_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def preprocess_dataset(dataset):
    return dataset


# ===========================
# FUNCTIONS FOR DUMMY EXPERIMENT
# ===========================
def get_dummy_dataloader():
    x_dummy = torch.randn(100, 10)  # 100 samples, 10 features
    # Generate binary labels (0 or 1) for 100 samples
    y_dummy = torch.randint(0, 2, (100, 1)).float()  # Use .float() for compatibility with BCELoss
    dataset = TensorDataset(x_dummy, y_dummy)
    dataloader = DataLoader(dataset, batch_size=10)
    
    return dataloader
