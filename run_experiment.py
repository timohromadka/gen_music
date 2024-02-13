import logging
import os
import traceback
from typing import Dict, List, Tuple
import sys
import wandb


import pytorch_lightning as pl
# torchaudio gives wierd errors
# use this thread to fix: https://github.com/pytorch/audio/issues/62
# pip install -U torch torchaudio --no-cache-dir
import torch
from torch.cuda import OutOfMemoryError

from args.args import parser, apply_subset_arguments
from models.models import DummyModel
from utils.wandb_utils import create_wandb_logger
from utils.dataset_utils import CustomDataModule, get_dataloaders, get_dummy_dataloader
from utils.training_utils import train_model



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Run Experiment Pipeline for Datamaps!')


def process_results(args):
    # If anything needs to be processed after training/testing, it should go here
    return


def run_dummy_experiment(wandb_logger):
    train_dataloader = get_dummy_dataloader()
    test_dataloader = get_dummy_dataloader() 

    model = DummyModel()
    trainer = pl.Trainer(max_epochs=3, logger=wandb_logger)

    trainer.fit(model, train_dataloader)
    trainer.test(model, dataloaders=[test_dataloader])


def main():
    try:
        # ================================
        # SET UP ARGS
        # ================================
        args = parser.parse_args()

        # ================================
        # CONFIGURE WANDB
        # ================================
        if args.disable_wandb:
            os.environ['WANDB_MODE'] = 'disabled'
        wandb.init(project=args.wandb_project_name, config=args)
    
        wandb_logger = create_wandb_logger(args, args.wandb_project_name)
        # wandb.run.name = f"{get_run_name(args)}_{args.suffix_wand_run_name}_{wandb.run.id}"
        wandb.run.name = args.wandb_run_name
        
        if args.run_dummy_experiment:
            run_dummy_experiment(wandb_logger)
        else:
            # ================================
            # FETCH DATASET 
            # ================================
            train_loader, train_unshuffled_loader, val_loader, test_loader = get_dataloaders(args)
            data_module = CustomDataModule(train_loader, val_loader, test_loader)
            
            # ================================
            # UNDERGO TRAINING
            # ================================
            train_model(
                args, data_module, train_unshuffled_loader, wandb_logger
            )
            
            process_results(args)
            
        # ================================
        # FINISH
        # ================================
        wandb.finish()
        

    except OutOfMemoryError as oom_error:
        # Log the error to wandb
        logger.warning(str(oom_error))
        wandb.log({"error": str(oom_error)})

        # Mark the run as failed
        wandb.run.fail()
        
        wandb.finish(exit_code=-1)
        

    except Exception as e:
        print(traceback.print_exc(), file=sys.stderr)
        print(f"An error occurred: {e}\n Terminating run here.")
        # Log error message to wandb
        wandb.log({"critical_error": str(e)})
        # Finish the wandb run without specifying exit_code if fail() is not available
        wandb.finish(exit_code=-1)

        
if __name__ == '__main__':
    main()
