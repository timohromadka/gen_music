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
from torchaudio.transforms import Spectrogram, MelSpectrogram, Resample
from tqdm import tqdm

from datasets import load_dataset, Audio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('utils/musiccaps_utils.py')

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

    logger.info('Beginning mapping process to extract youtube audios.')
    return ds.map(process, num_proc=num_proc, writer_batch_size=writer_batch_size, keep_in_memory=False).cast_column('audio', Audio(sampling_rate=sampling_rate))

# def preprocess_and_cache_musiccaps_dataset(args, cache_dir):
#     logger.info('Processing MusicCaps dataset...')
#     ds = process_and_download_musiccaps(
#         cache_dir,
#         sampling_rate=args.sample_rate,
#         limit=args.num_samples_for_train,
#         num_proc=args.num_gpus
#     )

#     processed_data = []
#     for idx, row in enumerate(ds):
#         base_file_path = os.path.join(cache_dir, f"{row['ytid']}")
#         wav_file_path = f"{base_file_path}.wav"
#         pt_file_path = f"{base_file_path}.pt"

#         # Ensure the subdirectory exists
#         os.makedirs(os.path.dirname(base_file_path), exist_ok=True)

#         # Skip if .pt file already exists
#         if os.path.exists(pt_file_path):
#             processed_data.append(pt_file_path)
#             continue

#         # Download and process WAV file if it doesn't exist
#         if not os.path.exists(wav_file_path):
#             success, log = download_clip(row['ytid'], wav_file_path, row['start_s'], row['end_s'])
#             if not success:
#                 logging.error(f'Failed to download or process file for ytid: {row["ytid"]}')
#                 continue

#         # Load, potentially resample, and save as .pt tensor
#         waveform, sample_rate = torchaudio.load(wav_file_path)
#         if sample_rate != args.sample_rate or args.dataset_type == 'waveform':
#             if sample_rate != args.sample_rate:
#                 resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=args.sample_rate)
#                 waveform = resampler(waveform)
#             torch.save((waveform, args.sample_rate), pt_file_path)
#             processed_data.append(pt_file_path)

#             # Optionally save resampled WAV file if requested
#             if args.save_wav_file and args.dataset_type == 'waveform':
#                 torchaudio.save(wav_file_path, waveform, args.sample_rate)
#         else:
#             # If no resampling is needed but saving as .pt is requested, do it here
#             torch.save((waveform, sample_rate), pt_file_path)
#             processed_data.append(pt_file_path)

#     return processed_data

def process_and_download_clip(example, cache_dir, sample_rate):
    base_file_path = os.path.join(cache_dir, example['ytid'])
    wav_file_path = f"{base_file_path}.wav"
    pt_file_path = f"{base_file_path}.pt"

    # Check if .pt file already exists to skip processing
    if os.path.exists(pt_file_path):
        return pt_file_path

    # Download WAV file if it doesn't exist
    if not os.path.exists(wav_file_path):
        success, log = download_clip(example['ytid'], wav_file_path, example['start_s'], example['end_s'])
        if not success:
            logging.error(f'Failed to download or process file for ytid: {example["ytid"]}')
            return None

    # Process WAV file
    waveform, original_sample_rate = torchaudio.load(wav_file_path)
    # if original_sample_rate != sample_rate:
    #     resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=sample_rate)
    #     waveform = resampler(waveform)

    # Save processed waveform as .pt tensor
    torch.save((waveform, sample_rate), pt_file_path)
    return pt_file_path

def preprocess_and_cache_musiccaps_dataset(args, cache_dir):
    logger.info('Processing MusicCaps dataset...')
    ds = load_dataset('google/MusicCaps', split='train')
    # these ids are no longer available on youtube
    ytid_to_remove = [
        '-sevczF5etI', 
        '0J_2K1Gvruk', 
        '0fqtA_ZBn_8',
        '0khKvVDyYV4',
        '0pewITE1550',
        '25Ccp8usBtE',
        '2dyxjGTXSpA',
        '374R7te0ra0',
        '5Y_mT93tkvQ',
        '63rqIYPHvlc',
        '7B1OAtD_VIA',
        '7LGLhQFiE0s',
        '7WZwlOrRELI',
        '8ZK1ajW598M',
        '8hT2G2ew6-c',
        '8olHAhUKkuk',
        'AFWy1qyyMHE',
        'AaUZb-iRStE',
        'Ah_aYOGnQ_I',
        'B7iRvj8y9aU',
        'BKQYrrJVg6g',
        'BeFzozm_H5M',
        'BiQik0xsWxk',
        'C7OIuhWSbjU',
        'CCFYOw8keiI',
        'Czbi1u-gwUU',
        'DHDn79ee98Q',
        'EhFWLbNBOxc',
        'FYwDTJtEzhk',
        'FtskdD6Py7Y',
        'Fv9swdLA-lo',
        'HAHn_zB47ig',
        'Hvs6Xwc6-gc',
        'IF-77lLlMzE',
        'IbJh1xeBFcI',
        'InfQUMh935c',
        'JNw0A8pRnsQ',
        'Jk2mvFrdZTU',
        'L5Uu_0xEZg4',
        'LRfVQsnaVQE',
        'MYtq46rNsCA',
        'NIcsJ8sEd0M',
        'OS4YFp3DiEE',
        'RQ0-sjpAPKU',
        'Rqv5fu1PCXA',
        'RxBFh-zdid4',
        'SLq-Co_szYo',
        'T6iv9GFIVyU',
        'TkclVqlyKx4',
        'UOC4VWQpnDM',
        'UdA6I_tXVHE',
        'UzAXqTsdtjY',
        'Vi2posMLsrg',
        'Vu7ZUUl4VPc',
        'WKYUiLace9Y',
        'We0WIPYrtRE',
        'WvEtOYCShfM',
        'XCQMJXkI5bA',
        'Xoke1wUwEXY',
        'Xy7KtmHMQzU',
        'ZLDQ8-8AyI4',
        'Zs3arUuPciY',
        '_ACyiKGpD8Y',
        '_DHMdtRRJzE',
        'ab8V7MYQVyg',
        'asYb6iDz_kM',
        'ba3QPheW8mI',
        'bpwO0lbJPF4',
        'cADT8fUucLQ',
        'd0Uz_RnRV88',
        'd6-bQMCz7j0',
        'dcY062mkf9g',
        'dsVzafZik3c',
        'eHeUipPZHIc',
        'ed-1zAOr9PQ',
        'fZyq2pM2-dI',
        'fwXh_lMOqu0',
        'g8USMvt9np0',
        'gdtw54I8soM',
        'go_7i6WvfeE',
        'hwp7wCKIXg4',
        'iXgEQj1Fs7g',
        'idVCpQaByc4',
        'itgeNVRhBKs',
        'j9hAUlz5kQs',
        'jd1IS7N3u0I',
        'jmPmqzxlOTY',
        'k-LkhT4HAiE',
        'keIvA2wSPZc',
        'kiu-40_T5nY',
        'lTLsL94ABRs',
        'lrk00BNiuD4',
        'm-e3w2yZ6sM',
        'nTtxF9Wyw6o',
        'opu2tXoVNKU',
        'qIxpfaeZ-zs',
        'qc1DaM4kdO0',
        'qyCea2TuUV4',
        'rQeikHMQGOM',
        'rfah0bhmYkc',
        't5fW1-6iXZY',
        'tEZU-ZRhSoY',
        'tju-t_Bz_W8',
        'tpamd6BKYU4',
        'vOAXAoHtl7o',
        'vQHKa69Mkzo',
        'vVNWjq9byoQ',
        'vsDriIwNAmU',
        'wBe5tW8iJew',
        'wSeT9dlfDTc',
        'xxCnmao8FAs',
        'yzT9nsHslAk',
        'z4ewUsnIJKI',
        'zCrpaLEq1VQ',
        'zSSIGv82318',
        'zSq2D_GF00o'
    ]
    ds = ds.filter(lambda example: example['ytid'] not in ytid_to_remove)


    # Limit the dataset if required
    if args.num_samples_for_train > 0:
        ds = ds.select(range(args.num_samples_for_train))

    processed_data = []
    for example in tqdm(ds, desc='Processing and downloading musiccaps clips.'):
        pt_file_path = process_and_download_clip(example, cache_dir, args.sample_rate)
        if pt_file_path:
            processed_data.append(pt_file_path)

    return processed_data