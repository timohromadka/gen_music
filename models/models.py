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

from .diffusion_models import AudioDiffusionLightningModule
from .gan_models import AudioGANLightningModule
from .vae_models import AudioVAELightningModule
from .dummy_models import DummyModel

import sys
sys.path.append('..')
from utils.constants import SAMPLE_LENGTH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('models/models.py')


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



def load_checkpoint(teacher_run_name, args):
    """
    Load a model checkpoint from a given checkpoint path.
    
    Args:
    - teacher_run_name (str): Run name of the teacher model.
    - args: Arguments needed to initialize the model architecture.
    
    Returns:
    - model checkpoint.
    """
    logger.info(f'Loading model checkpoint from run name: {teacher_run_name}')
    
    teacher_model_path = os.path.join(args.checkpoint_dir, teacher_run_name)
    
    if not os.path.exists(teacher_model_path):
        raise FileNotFoundError(f"Directory not found at {teacher_model_path}")

    checkpoint_files = glob.glob(os.path.join(teacher_model_path, '*.ckpt'))
    if not checkpoint_files:
        raise FileNotFoundError(f"No .ckpt files found in {teacher_model_path}")

    checkpoint_path = checkpoint_files[0]
    checkpoint = torch.load(checkpoint_path)
    
    return checkpoint


  
def load_model_from_run_name(args):
    """
    Load a model from a given checkpoint path.
    
    Args:
    - teacher_run_name (str): Run name of the teacher model.
    - args: Arguments needed to initialize the model architecture.
    
    Returns:
    - Loaded model.
    """
    logger.info(f'Loading, configuring, and initializing teacher model from checkpoint using run name: {args.run_name_to_load}')
    
    teacher_model_path = os.path.join(args.checkpoint_dir, args.run_name_to_load)
    
    if not os.path.exists(teacher_model_path):
        raise FileNotFoundError(f"Directory not found at {teacher_model_path}")

    checkpoint_files = glob.glob(os.path.join(teacher_model_path, '*.ckpt'))
    if not checkpoint_files:
        raise FileNotFoundError(f"No .ckpt files found in {teacher_model_path}")

    # There should only be one, if not, we only grab the first one for simplicity
    checkpoint_path = checkpoint_files[0]

    # Load the checkpoint to access the configuration
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    config = checkpoint.get('config')
    config_dict = vars(config)

    if not config:
        raise ValueError(f"No config found in checkpoint at {checkpoint_path}")
    
    model_args = copy.deepcopy(args)
    
    # dynamically copy all arguments from config to teacher_args
    # this is needed to ensure we can properly load and use the model for inference correctly
    for key, value in vars(config).items():
        setattr(model_args, key, value)

    if model_args.model == "diffusion":
        model_type = AudioDiffusionLightningModule
    elif model_args.model == 'vae':
        model_type =  AudioVAELightningModule
    elif model_args.model == 'gan':
        model_type = AudioGANLightningModule
    else:
        raise ValueError(f"Unsupported model type: {model_args.model}")

    model = model_type.load_from_checkpoint(checkpoint_path, args=model_args)

    return model, model_args