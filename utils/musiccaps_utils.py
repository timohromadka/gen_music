"""
The code for the function 'download_clip' was taken from the github repository:
https://github.com/nateraw/download-musiccaps-dataset

All credit belongs to the author for their code.
"""
import logging
import subprocess
import os
from pathlib import Path
import torch
import torchaudio

from datasets import load_dataset, Audio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('utils/dataset_utils.py')

def download_clip(
    video_identifier,
    output_filename,
    start_time,
    end_time,
    tmp_dir='/tmp/musiccaps',
    num_attempts=5,
    url_base='https://www.youtube.com/watch?v='
):
    status = False

    command = f"""
        yt-dlp --quiet --no-warnings -x --audio-format wav -f bestaudio -o "{output_filename}" --download-sections "*{start_time}-{end_time}" {url_base}{video_identifier}
    """.strip()

    attempts = 0
    while True:
        try:
            output = subprocess.check_output(command, shell=True,
                                                stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            attempts += 1
            if attempts == num_attempts:
                return status, err.output
        else:
            break

    # Check if the video was successfully saved.
    status = os.path.exists(output_filename)
    return status, 'Downloaded'


def process_and_download_musiccaps(
    data_dir: str, 
    sampling_rate: int = 44100, 
    limit: int = None, 
    num_proc: int = 1, 
    writer_batch_size: int = 1000
    ):
    ds = load_dataset('google/MusicCaps', split='train')
    if limit is not None:
        ds = ds.select(range(limit))

    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)

    def process(example):
        outfile_path = str(data_dir / f"{example['ytid']}.wav")
        status = True
        if not os.path.exists(outfile_path):
            status, log = download_clip(example['ytid'], outfile_path, example['start_s'], example['end_s'])
        example['audio'] = outfile_path
        example['download_status'] = status
        return example

    return ds.map(process, num_proc=num_proc, writer_batch_size=writer_batch_size, keep_in_memory=False).cast_column('audio', Audio(sampling_rate=sampling_rate))

def preprocess_and_cache_musiccaps_dataset(args, cache_dir):
    logging.info('Processing MusicCaps dataset...')
    ds = process_and_download_musiccaps(
        cache_dir,
        sampling_rate=args.sample_rate,
        limit=args.num_samples_for_train,
        num_proc=4
    )

    processed_data = []
    for idx, row in enumerate(ds):
        base_file_path = os.path.join(cache_dir, f"{row['ytid']}")
        wav_file_path = f"{base_file_path}.wav"
        pt_file_path = f"{base_file_path}.pt"

        # Ensure the subdirectory exists
        os.makedirs(os.path.dirname(base_file_path), exist_ok=True)

        # Skip if .pt file already exists
        if os.path.exists(pt_file_path):
            processed_data.append(pt_file_path)
            continue

        # Download and process WAV file if it doesn't exist
        if not os.path.exists(wav_file_path):
            success, log = download_clip(row['ytid'], wav_file_path, row['start_s'], row['end_s'])
            if not success:
                logging.error(f'Failed to download or process file for ytid: {row["ytid"]}')
                continue

        # Load, potentially resample, and save as .pt tensor
        waveform, sample_rate = torchaudio.load(wav_file_path)
        if sample_rate != args.sample_rate or args.dataset_type == 'waveform':
            if sample_rate != args.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=args.sample_rate)
                waveform = resampler(waveform)
            torch.save((waveform, args.sample_rate), pt_file_path)
            processed_data.append(pt_file_path)

            # Optionally save resampled WAV file if requested
            if args.save_wav_file and args.dataset_type == 'waveform':
                torchaudio.save(wav_file_path, waveform, args.sample_rate)
        else:
            # If no resampling is needed but saving as .pt is requested, do it here
            torch.save((waveform, sample_rate), pt_file_path)
            processed_data.append(pt_file_path)

    return processed_data