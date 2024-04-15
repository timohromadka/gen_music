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

from utils.musiccaps_utils import process_and_download_musiccaps
from utils.constants import SAMPLE_LENGTH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('utils/spotifysleepdataset_utils.py')

def get_spectrogram(waveform):
    spectrogram = Spectrogram(n_fft=400)(waveform)
    return spectrogram


def get_mel_spectrogram(waveform, sample_rate, n_mels):
    mel_spectrogram = MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)(waveform)
    return mel_spectrogram

def filter_df(df, dataset_name, n_samples=None):
    if dataset_name == 'spotify_sleep_dataset':
        if n_samples:
            df = df[:n_samples]
        df = df[df['SampleURL'].notna()]
        df = df.drop_duplicates(subset='TrackID', keep='first')
        
        excluded_genres = ['punk', 'funk', 'alternative', 'metal', 'electronic', 'house', 'r&b', 'rock', 'rap', 'pop', 'hip hop']
        pattern = '|'.join([f'(?i){genre}' for genre in excluded_genres])  # The (?i) makes the regex case-insensitive
        df = df[~df['Genres'].str.contains(pattern, na=False)]
    elif dataset_name == 'musiccaps':
        return df
    return df


def preprocess_and_cache_spotifysleep_dataset(args, cache_dir):
    # =========================
    # Get dataframe
    # =========================
    df = pd.read_csv('data/SPD_unique_withClusters.csv')
    logger.info(f'df.shape is: {df.shape}')
    logger.info(f'Filtering df...')
    df = filter_df(df, args.dataset, args.num_samples_for_train)
    logger.info(f'Done! df.shape is: {df.shape}.')

    processed_data = []
    file_extension = '.pt'
    
    # =========================
    # Fetch and cache data
    # =========================
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc='Fetching audio samples, performing necessary conversions, and caching.'):
        url = row['SampleURL']
        # print(f'Now fetching url: {url}')
        file_path = os.path.join(cache_dir, f"{idx}{file_extension}")
        # tensor_path = file_path.replace(file_extension, '_tensor.pt')  # Path for saving tensor if necessary

        # Ensure the subdirectory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        if not os.path.exists(file_path):
            response = requests.get(url)
            if response.status_code == 200:
                waveform, sample_rate = torchaudio.load(BytesIO(response.content))

                # Resample and convert to mono if required
                if args.num_channels == 1:
                    if waveform.size(0) > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    else:
                        raise ValueError("Cannot convert waveform to 2 channels since it only has 1 channel.")
                    
                if sample_rate != args.sample_rate:
                    resampler = Resample(orig_freq=sample_rate, new_freq=args.sample_rate)
                    waveform = resampler(waveform)

                if args.dataset_type == 'spectrogram':
                    waveform = get_spectrogram(waveform)
                elif args.dataset_type == 'mel-spectrogram':
                    waveform = get_mel_spectrogram(waveform, sample_rate, args.n_mels)

                # save as a .pt tensor
                torch.save((waveform, args.sample_rate), file_path)

            else:
                logger.error(f'Failed to download from url: {url}')
                continue
        processed_data.append(file_path)
        
        # ...and then as a .wav file (optional)
        wav_path = file_path.replace('.pt', '.wav')
        if not os.path.exists(wav_path):
            if args.save_wav_file and args.dataset_type == 'waveform':
                torchaudio.save(wav_path, waveform, args.sample_rate)

    return processed_data