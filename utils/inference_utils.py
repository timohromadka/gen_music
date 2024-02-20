import argparse
import os
import torch
from pathlib import Path
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler

def generate_samples(model, model_args, experiment_args, device):
    model.eval()
    samples = []
    with torch.no_grad():
        for _ in range(experiment_args.num_samples):
            noise = torch.randn(1, 2, model_args.sample_length).to(device)
            sample = model.sample(noise, num_steps=experiment_args.num_steps)
            samples.append(sample)
    return samples

def save_samples(samples, save_path):
    os.makedirs(save_path, exist_ok=True)
    for i, sample in enumerate(samples):
        torch.save(sample, os.path.join(save_path, f"sample_{i}.pt"))
