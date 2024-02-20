import torch
import torch.cuda as cuda
import torchaudio

from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler

def print_memory_usage():
    print(f"Current memory allocated: {cuda.memory_allocated()} bytes")
    print(f"Max memory allocated: {cuda.max_memory_allocated()} bytes")
    print(f"Current memory cached: {cuda.memory_reserved()} bytes")
    print(f"Max memory cached: {cuda.max_memory_reserved()} bytes")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print_memory_usage()
model = DiffusionModel(
    net_t=UNetV0, # The model type used for diffusion (U-Net V0 in this case)
    in_channels=2, # U-Net: number of input/output (audio) channels
    channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024], # U-Net: channels at each layer
    factors=[1, 4, 4, 4, 2, 2, 2, 2, 2], # U-Net: downsampling and upsampling factors at each layer
    items=[1, 2, 2, 2, 2, 2, 2, 4, 4], # U-Net: number of repeating items at each layer
    attentions=[0, 0, 0, 0, 0, 1, 1, 1, 1], # U-Net: attention enabled/disabled at each layer
    attention_heads=8, # U-Net: number of attention heads per attention item
    attention_features=64, # U-Net: number of attention features per attention item
    diffusion_t=VDiffusion, # The diffusion method used
    sampler_t=VSampler, # The diffusion sampler used
).to(device) # Move model to the appropriate device

print(f'Loaded model. \n\n')
print_memory_usage()

length = 2**17
batch_size = 8
# Train model with audio waveforms
# print_memory_usage()
# audio = torch.randn(batch_size, 2, length, device=device) # [batch_size, in_channels, length]
# print(audio)
# print(f'Loaded audio. \n\n')
# print_memory_usage()
# loss = model(audio)
# print(f'Calculated loss. \n\n')
# print_memory_usage()
# print(loss)
# loss.backward()
# print(f'Loss Backward. \n\n')
# print_memory_usage()
# print('Done')

# Turn noise into new audio sample with diffusion
noise = torch.randn(batch_size, 2, length, device=device) # [batch_size, in_channels, length]
print(f'Calculated noise. \n\n')
print('Sampling')
print_memory_usage()
sample = model.sample(noise, num_steps=10) # Suggested num_steps 10-100
print(f'Calculated sample. \n\n')
print_memory_usage()
print('done with sample')

sample = sample.squeeze(0)  # Assuming batch size of 1, adjust accordingly

# Save the audio sample
# You might need to adjust the sample_rate according to your specific use case
sample_rate = 44100  # Example sample rate, adjust as needed
torchaudio.save('generated_audio_2.wav', sample.cpu(), sample_rate)