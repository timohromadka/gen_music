import argparse
from datetime import datetime

# ======================================

parser = argparse.ArgumentParser(description='Arguments for Training and Sampling of Generative Audio Models.')

# Debugging
parser.add_argument('--run_dummy_experiment', required=False, action='store_true', help='If True, will run a dummy experiment (for wandb testing).')

# Experimental Procedure
parser.add_argument('--experiment_type', type=str, required=True, default='training', choices=['training', 'inference', 'evaluation', 'data_loading'])

# Inference
parser.add_argument("--num_batches_to_generate", type=int, default=100, help="Number of samples to generate")
parser.add_argument("--num_steps", type=int, default=100, help="Number of steps to sample for.")
parser.add_argument("--run_name_to_load", type=str, help="Specify the run_name, used for loading a model, or fetching a models arguments.")
parser.add_argument('--inference_batch_size', type=int, default=8, help='Batch size for inference.')
parser.add_argument("--run_name_to_load", type=str, help="Specify the run_name, used for loading a model, or fetching a models arguments.")

# Evaluation
parser.add_argument("--metrics", nargs='+', default=['Frechet Distance', 'Inception Score', 'Kullback-Leibler'], help="Metrics for evaluation")
parser.add_argument('--path_to_original_dataset', type=str, required=False, default='spotify_sleep_dataset', choices=['spotify_sleep_dataset', 'random'])

# Dataset
parser.add_argument('--dataset', type=str, required=False, default='spotify_sleep_dataset', choices=['spotify_sleep_dataset', 'random'])
parser.add_argument('--save_wav_file', required=False, action='store_true', help='If True, will save .wav files (waveform) for each data point in addition to .pt files.')
parser.add_argument('--num_samples_for_train', required=False, type=int, default=0, help='Specify how many samples to use from the dataset. If nothing is set, then full dataset will be used.')
parser.add_argument('--sample_length', required=False, type=int, help='Specify how long the samples should be, in seconds. If nothing is set, then the default of the dataset will be used.')
parser.add_argument('--trim_area', required=False, type=str, default='random', help='Specify where the audio trimming should happen. Choices are [random, start, end], default is random.')


# Model 
parser.add_argument('--model', type=str, required=True, choices=['diffusion', 'vae', 'gan'])
parser.add_argument('--model_size', type=str, default='tiny', required=True, choices=['tiny', 'small', 'medium', 'large'])
parser.add_argument('--pretrained', action='store_true', help='If True, use pretrained weights (usually pretrained on ImageNet).')

# Audio Setups
parser.add_argument('--sample_rate', type=int, default=24000)
parser.add_argument('--dataset_type', type=str, default='waveform', choices=['waveform', 'spectrogram', 'mel-spectrogram'])
parser.add_argument('--n_fft', type=int, default=400, help='Number of Fast Fourier Transform (FFT) points used for analyzing the audio signal. Basically, how many bins are we using to represent frequency spectrum?')
parser.add_argument('--n_mels', type=int, default=128, help='The number of Mel bands to generate in the Mel spectrogram. This controls the frequency resolution of the Mel spectrogram. Typically, a larger number of Mel bands provides finer frequency resolution. A common default value is 128, but this can be adjusted based on specific audio analysis requirements or computational constraints.')
parser.add_argument('--num_channels', type=int, help='Number of channels the audio waveform should have.')


# Training Configuration
parser.add_argument('--epochs', type=int, help='Maximum number of epochs for training.')
parser.add_argument('--min_steps', type=int, help='Minimum number of steps to train for before early stopping can occur (then monitored by patience_early_stopping).')
parser.add_argument('--train_by_epochs', action='store_true', help='If True, training will be done on a per-epoch basis (as defined by epochs). If false, training will be done on a per-step basis (as defined by max_steps).')
parser.add_argument('--max_steps', type=int, default=1000, help='Maximum number of training steps (batches) for training.')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'AdamW', 'SGD'], help='Optimizer for training.')
parser.add_argument('--train_batch_size', type=int, default=8, help='Batch size for training.')
parser.add_argument('--validation_batch_size', type=int, default=8, help='Batch size for validation.')
parser.add_argument('--gradient_clip_val', type=float, default=0.5, help='Gradient clipping value to prevent exploding gradients.')
parser.add_argument('--checkpoint_dir', type=str, default='./model_checkpoints/', help='Directory to save model checkpoints.')
parser.add_argument('--force_full_epoch_training', action='store_true', help='If True, then training will continue for the specified amount of epochs regardless.')
# PyTorch Lightning Specific
parser.add_argument('--precision', type=int, default=32, choices=[16, 32], help='Precision of training (32 or 16 for mixed precision).')
parser.add_argument('--accelerator', type=str, default='gpu', help='Type of accelerator to use ("gpu" or "cpu").')
parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use.')
parser.add_argument('--deterministic', action='store_true', help='Ensures reproducibility. May impact performance.')

# General Configuration
parser.add_argument('--cache_dir', type=str, default='cache/data', help='Specify where to store data (to repeat expensive processing/data fetching each run).')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
parser.add_argument('--data_augmentation', type=bool, default=True, help='Whether to use data augmentation.')
parser.add_argument('--logging_interval', type=int, default=100, help='Interval for logging training metrics.')
# parser.add_argument('--checkpoint_freq', type=int, default=3, help='Frequency of saving top-k model checkpoints.')
parser.add_argument('--save_top_k', type=int, default=1, help='Select k-best model checkpoints to save for each run.')
parser.add_argument('--test_only', action='store_true', help='If True, will only run testing/sampling, no training or validation will be performed.')
parser.add_argument('--val_split_seed', type=int, default=42, help='Seed for determining train/val/test split.')



# Validation
parser.add_argument('--metric_model_selection', type=str, default='val_loss',
                    choices=['cross_entropy_loss', 'total_loss', 'balanced_accuracy', 'accuracy', 'lr-Adam', 'train_loss', 'train_loss_step', 'train_acc', 'train_acc_step', 'val_loss', 'val_acc'], help='Metric used for model selection.')
parser.add_argument('--patience_early_stopping', type=int, default=3,
                    help='Set number of checks (set by *val_check_interval*) to do early stopping. Minimum training duration: args.val_check_interval * args.patience_early_stopping epochs')
parser.add_argument('--val_check_interval', type=float, default=1.0, 
                    help='Number of steps at which to check the validation. If set to 1.0, will simply perform the default behaviour of an entire batch before validation.')
parser.add_argument('--train_on_full_data', action='store_true', dest='train_on_full_data', \
                    help='Train on the full data (train + validation), leaving only `--test_split` for testing.')
parser.add_argument('--overfit_batches', type=int, default=0, help='PyTorch Lightning trick to pick only N batches and iteratively overfit on them. Useful for debugging. Default set to 0, i.e. normal behaviour.')


# Weights & Biases (wandb) Integration
parser.add_argument('--wandb_project_name', type=str, default='th716_mphil_project')
parser.add_argument('--wandb_run_name', type=str, default=f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
parser.add_argument('--wandb_log_freq', type=int, default=10)
parser.add_argument('--group', type=str, help="Group runs in wand")
parser.add_argument('--job_type', type=str, help="Job type for wand")
parser.add_argument('--notes', type=str, help="Notes for wandb logging.")
parser.add_argument('--tags', nargs='+', type=str, default=[], help='Tags for wandb')
parser.add_argument('--suffix_wand_run_name', type=str, default="", help="Suffix for run name in wand")
parser.add_argument('--wandb_log_model', action='store_true', dest='wandb_log_model', help='True for storing the model checkpoints in wandb')
parser.set_defaults(wandb_log_model=False)
parser.add_argument('--disable_wandb', action='store_true', dest='disable_wandb', help='True if you dont want to create wandb logs.')
parser.set_defaults(disable_wandb=False)

# ======================================
# arg checking utils
# ======================================
def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x} not a floating-point literal")

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError(f"{x} not in range [0,1]")
    return x

def apply_subset_arguments(subset_args_str, args):

    # Proceed only if the string is not empty
    if subset_args_str and subset_args_str is not None:
        # Split the subset argument string into individual arguments
        # Trim the string to remove any leading/trailing whitespace
        subset_args_str = subset_args_str.strip()
        subset_args = subset_args_str.split()
        
        # Iterate over the subset arguments and update the args Namespace
        i = 0
        while i < len(subset_args):
            arg = subset_args[i]
            # Ensure that it starts with '--'
            if arg.startswith("--"):
                key = arg[2:]  # Remove '--' prefix to match the args keys
                value = subset_args[i + 1]
                # Update the args Namespace if the attribute exists
                if hasattr(args, key):
                    # Convert value to the right type based on the existing attribute
                    attr_type = type(getattr(args, key))
                    setattr(args, key, attr_type(value))
                i += 2  # Move to the next argument
            else:
                raise ValueError(f"Expected an argument starting with '--', found: {arg}")

