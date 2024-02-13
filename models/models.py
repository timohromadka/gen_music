import copy
import logging
import os
import glob

from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score
import torch.nn as nn
import torchvision.models as models

import sys
sys.path.append('..')
from utils.constants import SAMPLE_LENGTH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('models/models.py')


class AudioDiffusionLightningModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        logger.info('Initializing diffusion model using DiffusionModel class.')
        # self.model = DiffusionModel(
        #     net_t=UNetV0,
        #     in_channels=1 if args.convert_to_mono else 2,
        #     channels=[8, 32, 64, 128, 256, 512, 512, 1024, 1024],
        #     factors=[1, 4, 4, 4, 2, 2, 2, 2, 2],
        #     items=[1, 2, 2, 2, 2, 2, 2, 4, 4],
        #     attentions=[0, 0, 0, 0, 0, 1, 1, 1, 1],
        #     attention_heads=8,
        #     attention_features=64,
        #     diffusion_t=VDiffusion,
        #     sampler_t=VSampler,
        # )
        self.model = DiffusionModel(
            net_t=UNetV0,
            in_channels=1,  # Assuming your audio data has 1 channel
            channels=[16, 32, 64, 128],  # Fewer and narrower layers
            factors=[1, 2, 2, 2],  # Simplified down/up-sampling factors
            items=[1, 1, 1, 1],  # Fewer items, simplifies architecture
            attentions=[0, 0, 0, 0],  # Disable attention or use it in fewer layers
            attention_heads=4,  # Fewer attention heads
            attention_features=32,  # Less features in attention layers
            diffusion_t=VDiffusion,
            sampler_t=VSampler,
        )
        logger.info('Done!')
        # Define metrics specific to audio models if needed

    def forward(self, audio):
        return self.model(audio)

    def training_step(self, batch, batch_idx):
        audio, _ = batch
        loss = self.model(audio)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
        
    def on_validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        audio, _ = batch
        loss = self.model(audio)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss}
        # Optionally perform sampling here and log or save samples
        # sample = self.model.sample(audio, num_steps=10)
        # Log or save the generated samples
        
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.log('avg_test_loss', avg_loss, on_epoch=True, prog_bar=True, logger=True)


    def test_step(self, batch, batch_idx):
        audio, _ = batch
        loss = self.model(audio)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # Additional test metrics or operations can go here
        return {'test_loss': loss}

        # Similar to validation_step, perform sampling and log or save samples

    def configure_optimizers(self):
        if self.args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        # Add other optimizers as needed
        else:
            raise ValueError("Unsupported optimizer type")

        return optimizer
    
class DummyModel(pl.LightningModule):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.linear = nn.Linear(10, 1)
        self.accuracy = Accuracy(task="binary", threshold=0.5)
        self.f1_score = F1Score(task="binary", num_classes=2, threshold=0.5)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y.int())
        f1 = self.f1_score(y_hat, y.int())
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_f1', f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'test_loss': loss, 'test_acc': acc, 'test_f1': f1}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
    
    
def get_model(args):
    logger.info(f'Fetching model. args.model {args.model}')

    if args.model == "diffusion":
        return AudioDiffusionLightningModule(args)
        
    elif args.model == 'vae':
        return AudioVAELightningModule(args)
    
    elif args.model == 'gan':
        return AudioGANLightningModule(args)
    
    else:
        raise ValueError(f"Invalid model type: {args.model}. Expected 'diffusion', 'vae', 'gan'.")

  
def load_model_from_run_name(teacher_run_name, args):
    """
    Load a model from a given checkpoint path.
    
    Args:
    - teacher_run_name (str): Run name of the teacher model.
    - args: Arguments needed to initialize the model architecture.
    
    Returns:
    - Loaded model.
    """
    logger.info(f'Loading, configuring, and initializing teacher model from checkpoint using teacher run name: {teacher_run_name}')
    
    teacher_model_path = os.path.join(args.checkpoint_dir, teacher_run_name)
    
    if not os.path.exists(teacher_model_path):
        raise FileNotFoundError(f"Directory not found at {teacher_model_path}")

    checkpoint_files = glob.glob(os.path.join(teacher_model_path, '*.ckpt'))
    if not checkpoint_files:
        raise FileNotFoundError(f"No .ckpt files found in {teacher_model_path}")

    # There should only be one, if not, we only grab the first one for simplicity
    checkpoint_path = checkpoint_files[0]

    # Load the checkpoint to access the configuration
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint.get('config')

    if not config:
        raise ValueError(f"No config found in checkpoint at {checkpoint_path}")

    # Determine the model class based on the configuration
    model_type = config.get('model_type')
    model_size = config.get('model_size')
    pretrained_from_github = config.get('pretrained_from_github')
    pretrained = config.get('pretrained')
    
    # Create a new copy of args for the teacher model
    teacher_args = copy.deepcopy(args)
    teacher_args.model_size = model_size
    teacher_args.pretrained_from_github = pretrained_from_github
    teacher_args.pretrained = pretrained

    if model_type == "diffusion":
        model = AudioDiffusionLightningModule
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model = model.load_from_checkpoint(checkpoint_path, args=teacher_args)

    return model