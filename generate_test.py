import torch
import numpy as np
import os
import torchaudio
import argparse
from tqdm import tqdm

# Parse command line arguments
parser = argparse.ArgumentParser(description="Generate sine wave audio files.")
parser.add_argument('--freq_base', type=int, default=440, help='Base frequency for the first sine wave sample.')
parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate in Hz.')
parser.add_argument('--duration', type=int, default=1, help='Duration in seconds.')
parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to generate.')
parser.add_argument('--is_different', action='store_true', help='Generate completely different sets of audio files.')
args = parser.parse_args()

# Function to generate a sine wave
def generate_sine_wave(freq, sample_rate, duration):
    t = torch.linspace(0, duration, int(sample_rate * duration))
    waveform = torch.sin(2 * np.pi * freq * t)
    return waveform

# Directories for saving the waveforms
reference_dir = 'data/test_reference'
generated_dir = 'data/test_generated'

# Create directories if they don't exist
os.makedirs(reference_dir, exist_ok=True)
os.makedirs(generated_dir, exist_ok=True)

# Generate and save sine wave samples
for i in tqdm(range(args.num_samples)):
    freq = args.freq_base + i * 10  # Slightly different frequency for each sample
    
    # Generate sine wave for reference
    waveform = generate_sine_wave(freq, args.sample_rate, args.duration)
    # Save to reference directory in .wav format
    torchaudio.save(os.path.join(reference_dir, f'sample_{i}.wav'), waveform.unsqueeze(0), args.sample_rate)
    
    if args.is_different:
        # Generate a significantly different sine wave for generated directory
        # For example, using a much higher frequency base
        freq_generated = freq + 1000  # Adjust this value as needed for your test
    else:
        # Generate another sine wave with the same parameters
        freq_generated = freq
    
    waveform_generated = generate_sine_wave(freq_generated, args.sample_rate, args.duration)
    # Save to generated directory in .wav format
    torchaudio.save(os.path.join(generated_dir, f'sample_{i}.wav'), waveform_generated.unsqueeze(0), args.sample_rate)

print("Waveform generation and saving completed.")
