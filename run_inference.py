import argparse
import logging
import os
import torch
from pathlib import Path
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler

from args.inference_args import parser, apply_subset_arguments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Run Sampling/Inference for Generative Audio Models!')

def generate_samples(model, num_samples, device, sample_length=2**18, num_steps=10):
    model.eval()
    samples = []
    with torch.no_grad():
        for _ in range(num_samples):
            noise = torch.randn(1, 2, sample_length).to(device)
            sample = model.sample(noise, num_steps=num_steps)
            samples.append(sample)
    return samples

def save_samples(samples, save_path):
    os.makedirs(save_path, exist_ok=True)
    for i, sample in enumerate(samples):
        torch.save(sample, os.path.join(save_path, f"sample_{i}.pt"))

def main():
    args = parser.parse_args()

    # Initialize model here. Adjust according to your saved model loading logic if necessary
    model = DiffusionModel(
        net_t=UNetV0,
        in_channels=2,
        channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024],
        factors=[1, 4, 4, 4, 2, 2, 2, 2, 2],
        items=[1, 2, 2, 2, 2, 2, 2, 4, 4],
        attentions=[0, 0, 0, 0, 0, 1, 1, 1, 1],
        attention_heads=8,
        attention_features=64,
        diffusion_t=VDiffusion,
        sampler_t=VSampler,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load model checkpoint if necessary here
    # model.load_state_dict(torch.load(f"model_checkpoints/{args.run_name}/model.ckpt"))

    samples = generate_samples(model, args.num_samples, device)
    save_path = f"data/{args.run_name}/generated_samples/"
    save_samples(samples, save_path)

    print(f"Generated and saved {args.num_samples} samples to {save_path}")

if __name__ == "__main__":
    main()
