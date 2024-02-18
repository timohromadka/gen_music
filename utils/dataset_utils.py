import glob
import logging
from io import BytesIO
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

from utils.constants import SAMPLE_LENGTH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('utils/dataset_utils.py')

def get_spectrogram(waveform):
    spectrogram = Spectrogram(n_fft=400)(waveform)
    return spectrogram


def get_mel_spectrogram(waveform, sample_rate, n_mels):
    mel_spectrogram = MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)(waveform)
    return mel_spectrogram



class AudioDataset(Dataset):
    def __init__(self, df, args):
        self.dataset_type = args.dataset_type
        self.cache_dir = os.path.join(args.cache_dir, args.dataset, self.dataset_type)
        self.sample_rate = args.sample_rate
        self.sample_length = SAMPLE_LENGTH[args.dataset] if not args.sample_length else args.sample_length
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_file = os.path.join(self.cache_dir, f"{self.dataset_type}_data.pth")

        if os.path.exists(cache_file):
            self.data_paths = torch.load(cache_file)
        else:
            self.data_paths = preprocess_and_cache_dataset(df, args)
            torch.save(self.data_paths, cache_file)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        file_path = self.data_paths[idx]
        if os.path.exists(file_path):
            waveform, sample_rate = torch.load(file_path)
            # Ensure the waveform is of the expected size, trim if necessary
            expected_size = self.sample_length * self.sample_rate  # Set this to your desired size
            waveform = waveform[:, :expected_size]
            return waveform, sample_rate
        else:
            logger.error(f'File not found: {file_path}')
            return None, None



def filter_df(df, n_samples):
    if n_samples:
        df = df[:n_samples]
    df = df[df['SampleURL'].notna()]
    df = df.drop_duplicates(subset='TrackID', keep='first')
    return df


def preprocess_and_cache_dataset(df, args):
    logger.info(f'df.shape is: {df.shape}')
    logger.info(f'Filtering df')
    df = filter_df(df, args.n_samples)
    logger.info(f'Done! df.shape is: {df.shape}.')

    processed_data = []
    
    # define constants
    file_extension = '.wav'
    
    # check if data has already been cached!
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc='Fetching audio samples, performing necessary conversions, and caching.'):
        # Construct the file paths with dataset and data type subdirectories
        url = row['SampleURL']
        # print(f'Now fetching url: {url}')
        file_name = f"{idx}{file_extension}"
        file_path = os.path.join(args.cache_dir, args.dataset, args.dataset_type, file_name)
        tensor_path = file_path.replace(file_extension, '_tensor.pt')  # Path for saving tensor if necessary

        # Ensure the subdirectory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        if not os.path.exists(file_path):
            response = requests.get(url)
            if response.status_code == 200:
                waveform, sample_rate = torchaudio.load(BytesIO(response.content))

                # Resample and convert to mono if required
                # TODO
                # Make sure this iss actually doing what we believe its doing!????
                if sample_rate != args.sample_rate:
                    resampler = Resample(orig_freq=sample_rate, new_freq=args.sample_rate)
                    waveform = resampler(waveform)
                if args.convert_to_mono and waveform.size(0) > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                if args.dataset_type == 'spectrogram':
                    waveform = get_spectrogram(waveform)
                elif args.dataset_type == 'mel-spectrogram':
                    waveform = get_mel_spectrogram(waveform, sample_rate, args.n_mels)

                # save as a .pt tensor
                torch.save((waveform, sample_rate), tensor_path)
                # ...and then as a .wav file (optional)
                if args.save_wav_file and args.dataset_type == 'waveform':
                    torchaudio.save(file_path.replace('_tensor.pt', '.wav'), waveform, args.sample_rate)
                file_path = tensor_path  # Update file_path to point to the tensor file

            else:
                logger.error(f'Failed to download from url: {url}')
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


def get_train_val_test_sets(args):
    logger.info(f'Initializing dataset: {args.dataset}')

    if args.dataset == 'spotify_sleep_dataset':
        df = pd.read_csv('data/SPD_unique_withClusters.csv')
    else:
        raise ValueError("Unknown dataset_name")

    logger.info(f'Creating AudioDataset')
    dataset = AudioDataset(df, args)

    logger.info(f'Splitting into train, validation, and test.')
    # Split the dataset into 80% train and 20% temp (to be split further into validation and test)
    train_size = int(0.8 * len(dataset))
    temp_size = len(dataset) - train_size
    train_dataset, temp_dataset = random_split(dataset, [train_size, temp_size])

    # Split the temp dataset into validation and test (50% each of temp dataset, which is 10% each of the original dataset)
    val_size = test_size = int(temp_size / 2)
    local_generator = torch.Generator().manual_seed(args.val_split_seed)
    val_dataset, test_dataset = random_split(temp_dataset, [val_size, test_size], generator=local_generator)

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

    return train_loader, train_unshuffled_loader, val_loader, test_loader

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