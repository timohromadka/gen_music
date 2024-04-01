import argparse
import os
import torch
from pathlib import Path
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler

from tqdm import tqdm

from .dataset_utils import get_trimmed_waveform, calculate_waveform_length

def generate_samples(model, model_args, experiment_args, device, samples_save_dir):
    model.eval()
    
    waveform_length = calculate_waveform_length(model_args.sample_length, model_args.sample_rate)
    if model_args.model == 'diffusion':
        with torch.no_grad():
            for i in tqdm(range(experiment_args.num_batches_to_generate), desc='generating samples'):
                noise = torch.randn(experiment_args.inference_batch_size, model_args.num_channels, waveform_length).to(device)
                batch_samples = model.sample(noise, experiment_args.num_steps_for_inference)
                
                for j, sample in enumerate(batch_samples):
                    sample_file_path = os.path.join(samples_save_dir, f'sample_{i*experiment_args.inference_batch_size+j}.pt')
                    torch.save(sample.cpu(), sample_file_path)
        
    elif model_args.model == 'vae':
        # Implement VAE sampling and saving logic here
        pass
    
    elif model_args.model == 'gan':
        # Implement GAN sampling and saving logic here
        pass
    

def save_samples(samples, save_path):
    os.makedirs(save_path, exist_ok=True)
    for i, sample in enumerate(samples):
        torch.save(sample, os.path.join(save_path, f"sample_{i}.pt"))
