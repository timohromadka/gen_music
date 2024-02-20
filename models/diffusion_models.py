import logging

from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
import pytorch_lightning as pl
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('models/diffusion_models.py')

class AudioDiffusionLightningModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.on_step = not args.train_by_epochs
        self.on_epoch = args.train_by_epochs
        logger.info('Initializing diffusion model using DiffusionModel class.')

        # Adjustments for "large" and "medium" to have 8 layers, "small" and "tiny" to have 4 layers
        if args.model_size == 'large':
            channels = [32, 64, 128, 256, 512, 512, 1024, 1024]  # 8 layers
            factors = [1, 2, 2, 2, 2, 2, 2, 2]
            items = [1, 1, 2, 2, 2, 4, 4, 4]
            attentions = [0, 0, 1, 1, 1, 1, 1, 1]
            attention_heads = 8
            attention_features = 64
        elif args.model_size == 'medium':
            channels = [16, 32, 64, 128, 256, 256, 512, 512]  # 8 layers
            factors = [1, 2, 2, 2, 2, 2, 2, 2]
            items = [1, 1, 1, 2, 2, 2, 4, 4]
            attentions = [0, 0, 0, 1, 1, 1, 1, 1]
            attention_heads = 4
            attention_features = 32
        elif args.model_size == 'small':
            channels = [16, 32, 64, 128]  # 4 layers
            factors = [1, 2, 2, 2]
            items = [1, 1, 1, 2]
            attentions = [0, 0, 0, 1]
            attention_heads = 2
            attention_features = 16
        elif args.model_size == 'tiny':
            channels = [8, 16, 32, 64]  # 4 layers
            factors = [1, 2, 2, 2]
            items = [1, 1, 1, 1]
            attentions = [0, 0, 0, 0]
            attention_heads = 2
            attention_features = 16
        else:
            raise ValueError(f"Unsupported model size: {args.model_size}")

        self.model = DiffusionModel(
            net_t=UNetV0,
            in_channels=1 if args.convert_to_mono else 2,
            channels=channels,
            factors=factors,
            items=items,
            attentions=attentions,
            attention_heads=attention_heads,
            attention_features=attention_features,
            diffusion_t=VDiffusion,
            sampler_t=VSampler,
        )
        logger.info('Done!')
        # Define metrics specific to audio models if needed

    def forward(self, audio):
        return self.model(audio)

    def training_step(self, batch, batch_idx):
        audio, _ = batch
        loss = self.model(audio) # loss calculation handled in Diffusion object's forward method
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss_step', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss
        
    # def on_validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     self.log('avg_val_loss', avg_loss, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        audio, _ = batch
        loss = self.model(audio)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss}
        # Optionally perform sampling here and log or save samples
        # sample = self.model.sample(audio, num_steps=10)
        # Log or save the generated samples
        
    # def test_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
    #     self.log('avg_test_loss', avg_loss, on_epoch=True, prog_bar=True, logger=True)


    def test_step(self, batch, batch_idx):
        audio, _ = batch
        loss = self.model(audio)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        # self.log('test_loss', loss, on_step=self.on_step, on_epoch=self.on_epoch, prog_bar=True, logger=True)
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