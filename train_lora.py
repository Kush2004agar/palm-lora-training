"""
Main LoRA training script for palm dataset
"""

import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model_state_dict
import argparse
import json
import logging
from pathlib import Path
import yaml
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import wandb

try:
    from torch_fidelity import calculate_metrics
except ImportError:
    calculate_metrics = None

class TrainingConfig:
    """Training configuration"""
    def __init__(self):
        # Model
        self.pretrained_model = "runwayml/stable-diffusion-v1-5"
        
        # Paths
        self.data_dir = "./palm_dataset/10_palm"
        self.output_dir = "./output"
        self.logging_dir = "./logs"
        
        # Training
        self.resolution = 512
        self.batch_size = 4
        self.num_epochs = 15
        self.learning_rate = 1e-4
        self.lr_scheduler = "cosine"
        self.lr_warmup_steps = 500
        self.gradient_accumulation_steps = 1
        self.max_grad_norm = 1.0
        self.validation_split = 0.1
        self.eval_every_n_epochs = 1
        
        # LoRA
        self.lora_rank = 16
        self.lora_alpha = 16
        self.lora_dropout = 0.0
        self.lora_target_modules = ["to_k", "to_q", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"]
        
        # Saving
        self.save_every_n_epochs = 2
        self.save_model_as = "safetensors"  # or "pt"
        self.keep_top_k = 3
        
        # Other
        self.mixed_precision = "fp16"  # or "bf16" or "no"
        self.seed = 42
        self.num_workers = 4
        self.use_wandb = False
        self.use_tensorboard = True
        self.log_level = "INFO"
        self.use_ema = True
        self.ema_decay = 0.999
        self.min_snr_gamma = 5.0
        self.enable_latent_cache = True
        self.latent_cache_dir = "./output/latent_cache"
        self.fid_num_images = 500
        
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

class PalmDataset(Dataset):
    """Palm image dataset"""
    def __init__(self, data_dir, tokenizer, resolution=512, augment=False):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.augment = augment
        
        # Get all image files
        self.image_paths = [
            os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5) if augment else transforms.Lambda(lambda x: x),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1) if augment else transforms.Lambda(lambda x: x),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)) if augment else transforms.Lambda(lambda x: x),
            transforms.RandomResizedCrop(size=resolution, scale=(0.9, 1.0), interpolation=transforms.InterpolationMode.BICUBIC) if augment else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        
        print(f"📊 Dataset loaded: {len(self.image_paths)} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        txt_path = img_path.rsplit('.', 1)[0] + '.txt'
        
        # Load and process image
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Load caption
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                caption = f.read().strip()
        else:
            caption = "palm"
        
        # Tokenize caption
        input_ids = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids[0]
        
        return {
            "pixel_values": image,
            "input_ids": input_ids
        }


class LatentCacheDataset(Dataset):
    """Dataset that serves cached latents + tokenized prompts."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class EMA:
    """Exponential moving average for UNet weights."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

class LoRATrainer:
    """LoRA Trainer"""
    def __init__(self, config):
        self.config = config
        self.accelerator = None
        self.models = {}
        self.writer = None
        self.logger = logging.getLogger("lora-trainer")
        self.best_fid = float("inf")
        self.top_checkpoints = []
        self.ema = None
        
    def setup(self):
        """Setup training environment"""
        print("🔧 Setting up training environment...")
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(message)s",
        )
        
        # Set seed
        set_seed(self.config.seed)
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            log_with="wandb" if self.config.use_wandb else None
        )
        
        # Create output directories
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.logging_dir, exist_ok=True)
        
        # Initialize wandb
        if self.config.use_wandb and self.accelerator.is_main_process:
            wandb.init(
                project="palm-lora-training",
                config=self.config.to_dict()
            )

        if self.config.use_tensorboard and self.accelerator.is_main_process:
            self.writer = SummaryWriter(log_dir=self.config.logging_dir)
    
    def load_models(self):
        """Load pre-trained models"""
        print("📦 Loading models...")
        
        # Load tokenizer and text encoder
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config.pretrained_model, 
            subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            self.config.pretrained_model, 
            subfolder="text_encoder"
        )
        
        # Load VAE
        vae = AutoencoderKL.from_pretrained(
            self.config.pretrained_model, 
            subfolder="vae"
        )
        
        # Load UNet
        unet = UNet2DConditionModel.from_pretrained(
            self.config.pretrained_model, 
            subfolder="unet"
        )
        
        # Freeze VAE and text encoder
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        
        # Add LoRA to UNet
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
        )
        
        # Inject LoRA
        from peft import inject_adapter_in_model
        unet = inject_adapter_in_model(lora_config, unet)
        
        # Count parameters
        trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in unet.parameters())
        
        print(f"✅ Trainable parameters: {trainable_params:,} ({trainable_params/all_params*100:.2f}%)")
        
        self.models = {
            'vae': vae,
            'text_encoder': text_encoder,
            'unet': unet
        }
        
        return self.tokenizer
    
    def create_dataloaders(self, tokenizer):
        """Create train/validation dataloaders."""
        print("📂 Creating dataloader...")
        def seed_worker(worker_id):
            worker_seed = self.config.seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        
        full_dataset = PalmDataset(
            self.config.data_dir,
            tokenizer,
            self.config.resolution,
            augment=True,
        )

        val_size = max(1, int(len(full_dataset) * self.config.validation_split))
        train_size = max(1, len(full_dataset) - val_size)
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.seed),
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(self.config.seed),
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(self.config.seed),
        )

        self.logger.info("Train size: %d | Val size: %d", train_size, val_size)
        return train_dataloader, val_dataloader

    def _snr_weighted_loss(self, noise_scheduler, timesteps, model_pred, target):
        """Min-SNR loss re-weighting for diffusion stability."""
        mse = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        mse = mse.mean(dim=(1, 2, 3))
        alphas_cumprod = noise_scheduler.alphas_cumprod.to(model_pred.device)
        alpha_t = alphas_cumprod[timesteps]
        snr = alpha_t / (1 - alpha_t)
        gamma = self.config.min_snr_gamma
        weights = torch.minimum(snr, torch.full_like(snr, gamma)) / snr
        return (weights * mse).mean()

    def _cache_or_build_latents(self, vae, text_encoder, dataloader):
        """Precompute latents for faster epochs."""
        def seed_worker(worker_id):
            worker_seed = self.config.seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        if not self.config.enable_latent_cache:
            return dataloader

        os.makedirs(self.config.latent_cache_dir, exist_ok=True)
        cache_file = os.path.join(self.config.latent_cache_dir, "latents.pt")
        if os.path.exists(cache_file):
            samples = torch.load(cache_file, map_location="cpu")
            self.logger.info("Loaded cached latents: %d samples", len(samples))
            return DataLoader(
                LatentCacheDataset(samples),
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=True,
                drop_last=True,
                worker_init_fn=seed_worker,
                generator=torch.Generator().manual_seed(self.config.seed),
            )

        self.logger.info("Building latent cache...")
        samples = []
        vae.eval()
        text_encoder.eval()
        for batch in tqdm(dataloader, desc="Caching latents", disable=not self.accelerator.is_local_main_process):
            with torch.no_grad():
                pixel_values = batch["pixel_values"].to(self.accelerator.device)
                input_ids = batch["input_ids"].to(self.accelerator.device)
                latents = vae.encode(pixel_values).latent_dist.mode()
                latents = latents * vae.config.scaling_factor
                embeds = text_encoder(input_ids)[0]
            for i in range(latents.size(0)):
                samples.append(
                    {
                        "latents": latents[i].detach().cpu(),
                        "encoder_hidden_states": embeds[i].detach().cpu(),
                    }
                )

        torch.save(samples, cache_file)
        self.logger.info("Latent cache saved: %s", cache_file)
        return DataLoader(
            LatentCacheDataset(samples),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(self.config.seed),
        )

    def _compute_validation_loss(self, unet, val_dataloader, noise_scheduler):
        unet.eval()
        losses = []
        with torch.no_grad():
            for batch in val_dataloader:
                latents = self.models["vae"].encode(batch["pixel_values"].to(self.accelerator.device)).latent_dist.mode()
                latents = latents * self.models["vae"].config.scaling_factor
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = self.models["text_encoder"](batch["input_ids"].to(self.accelerator.device))[0]
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                target = noise if noise_scheduler.config.prediction_type == "epsilon" else noise_scheduler.get_velocity(latents, noise, timesteps)
                loss = self._snr_weighted_loss(noise_scheduler, timesteps, model_pred, target)
                losses.append(loss.item())
        unet.train()
        return float(np.mean(losses)) if losses else 0.0

    def _compute_fid(self):
        """Compute FID if torch-fidelity is installed."""
        if calculate_metrics is None:
            self.logger.warning("torch-fidelity not installed; skipping FID.")
            return None

        gen_dir = os.path.join(self.config.output_dir, "fid_samples")
        os.makedirs(gen_dir, exist_ok=True)
        for p in Path(gen_dir).glob("*.png"):
            p.unlink()

        pipe = self._build_eval_pipeline()
        prompt = "palm"
        with torch.no_grad():
            for i in range(self.config.fid_num_images):
                image = pipe(prompt=prompt, num_inference_steps=30, guidance_scale=7.0).images[0]
                image.save(os.path.join(gen_dir, f"{i:06d}.png"))

        metrics = calculate_metrics(
            input1=self.config.data_dir,
            input2=gen_dir,
            cuda=torch.cuda.is_available(),
            fid=True,
            verbose=False,
        )
        return float(metrics["frechet_inception_distance"])

    def _build_eval_pipeline(self):
        from diffusers import StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(
            self.config.pretrained_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,
        ).to(self.accelerator.device)
        ckpt_dir = os.path.join(self.config.output_dir, "checkpoint-final")
        if os.path.isdir(ckpt_dir):
            pipe.load_lora_weights(ckpt_dir)
        return pipe

    def _update_top_k(self, checkpoint_dir, fid_value):
        self.top_checkpoints.append((fid_value, checkpoint_dir))
        self.top_checkpoints.sort(key=lambda x: x[0])
        while len(self.top_checkpoints) > self.config.keep_top_k:
            _, drop_dir = self.top_checkpoints.pop()
            if os.path.isdir(drop_dir):
                for p in Path(drop_dir).glob("*"):
                    p.unlink()
                os.rmdir(drop_dir)
    
    def train(self):
        """Main training loop"""
        print("\n🎯 Starting training...\n")
        
        # Setup
        self.setup()
        tokenizer = self.load_models()
        train_dataloader, val_dataloader = self.create_dataloaders(tokenizer)
        
        # Get models
        vae = self.models['vae']
        text_encoder = self.models['text_encoder']
        unet = self.models['unet']
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            [p for p in unet.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-8
        )
        
        # Learning rate scheduler
        from diffusers.optimization import get_scheduler
        lr_scheduler = get_scheduler(
            self.config.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.config.lr_warmup_steps,
            num_training_steps=len(train_dataloader) * self.config.num_epochs
        )
        
        # Noise scheduler
        noise_scheduler = DDPMScheduler.from_pretrained(
            self.config.pretrained_model, 
            subfolder="scheduler"
        )
        
        # Prepare with accelerator
        unet, optimizer, train_dataloader, val_dataloader, lr_scheduler = self.accelerator.prepare(
            unet, optimizer, train_dataloader, val_dataloader, lr_scheduler
        )
        
        # Move to device
        vae.to(self.accelerator.device)
        text_encoder.to(self.accelerator.device)
        if self.config.use_ema:
            self.ema = EMA(unet, decay=self.config.ema_decay)

        train_dataloader = self._cache_or_build_latents(vae, text_encoder, train_dataloader)
        if self.config.enable_latent_cache:
            train_dataloader = self.accelerator.prepare(train_dataloader)
        
        # Training loop
        global_step = 0
        
        for epoch in range(self.config.num_epochs):
            unet.train()
            epoch_loss = 0
            
            progress_bar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
                disable=not self.accelerator.is_local_main_process
            )
            
            for step, batch in enumerate(progress_bar):
                with self.accelerator.accumulate(unet):
                    if self.config.enable_latent_cache:
                        latents = batch["latents"].to(self.accelerator.device)
                        encoder_hidden_states = batch["encoder_hidden_states"].to(self.accelerator.device)
                    else:
                        with torch.no_grad():
                            latents = vae.encode(batch["pixel_values"].to(self.accelerator.device)).latent_dist.mode()
                            latents = latents * vae.config.scaling_factor
                            encoder_hidden_states = text_encoder(
                                batch["input_ids"].to(self.accelerator.device)
                            )[0]

                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    
                    # Sample timesteps
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,),
                        device=latents.device
                    ).long()
                    
                    # Add noise to latents
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    
                    # Predict noise
                    model_pred = unet(
                        noisy_latents, 
                        timesteps, 
                        encoder_hidden_states
                    ).sample
                    
                    # Calculate loss
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                    
                    loss = self._snr_weighted_loss(noise_scheduler, timesteps, model_pred, target)
                    
                    # Backprop
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            [p for p in unet.parameters() if p.requires_grad],
                            self.config.max_grad_norm
                        )
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    if self.config.use_ema and self.accelerator.sync_gradients:
                        self.ema.update(unet)
                
                # Update progress
                epoch_loss += loss.detach().item()
                global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.detach().item():.4f}',
                    'lr': f'{lr_scheduler.get_last_lr()[0]:.2e}'
                })
                
                # Log to wandb
                if self.config.use_wandb and self.accelerator.is_main_process:
                    wandb.log({
                        "loss": loss.detach().item(),
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "step": global_step
                    })
                if self.writer and self.accelerator.is_main_process:
                    self.writer.add_scalar("train/loss", loss.detach().item(), global_step)
                    self.writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], global_step)
            
            # Epoch complete
            avg_loss = epoch_loss / len(train_dataloader)
            print(f"\n📊 Epoch {epoch+1} - Average Loss: {avg_loss:.4f}\n")
            val_loss = self._compute_validation_loss(unet, val_dataloader, noise_scheduler)
            self.logger.info("Epoch %d validation loss: %.4f", epoch + 1, val_loss)
            if self.writer and self.accelerator.is_main_process:
                self.writer.add_scalar("val/loss", val_loss, epoch + 1)
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                ckpt_dir = self.save_checkpoint(
                    unet, optimizer, lr_scheduler, epoch + 1, global_step, avg_loss, val_loss
                )
                if (epoch + 1) % self.config.eval_every_n_epochs == 0:
                    fid_value = self._compute_fid()
                    if fid_value is not None:
                        self.logger.info("Epoch %d FID: %.3f", epoch + 1, fid_value)
                        if self.writer and self.accelerator.is_main_process:
                            self.writer.add_scalar("eval/fid", fid_value, epoch + 1)
                        self._update_top_k(ckpt_dir, fid_value)
        
        # Save final model
        print("\n💾 Saving final model...")
        self.save_checkpoint(unet, optimizer, lr_scheduler, "final", global_step, avg_loss, val_loss)
        
        print("\n✅ Training complete!")
        
        if self.config.use_wandb:
            wandb.finish()
        if self.writer:
            self.writer.close()
    
    def save_checkpoint(self, unet, optimizer, lr_scheduler, epoch, global_step, train_loss, val_loss):
        """Save model checkpoint"""
        save_dir = os.path.join(self.config.output_dir, f"checkpoint-{epoch}")
        os.makedirs(save_dir, exist_ok=True)
        
        # Get LoRA state dict
        unet = self.accelerator.unwrap_model(unet)
        lora_state_dict = get_peft_model_state_dict(unet)
        
        # Save
        if self.config.save_model_as == "safetensors":
            from safetensors.torch import save_file
            save_file(lora_state_dict, os.path.join(save_dir, "pytorch_lora_weights.safetensors"))
        else:
            torch.save(lora_state_dict, os.path.join(save_dir, "pytorch_lora_weights.bin"))
        
        # Save config
        with open(os.path.join(save_dir, "training_config.json"), 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)

        training_state = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "best_fid": self.best_fid,
        }
        torch.save(training_state, os.path.join(save_dir, "training_state.pt"))
        
        print(f"💾 Saved checkpoint to {save_dir}")
        return save_dir

def main():
    parser = argparse.ArgumentParser(description="Train LoRA on palm dataset")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--pretrained_model", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--validation_split", type=float, default=None)
    parser.add_argument("--eval_every_n_epochs", type=int, default=None)
    parser.add_argument("--save_every_n_epochs", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--lr_warmup_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log_level", type=str, default=None)
    parser.add_argument("--disable_tensorboard", action="store_true")
    parser.add_argument("--disable_ema", action="store_true")
    parser.add_argument("--enable_latent_cache", action="store_true")
    parser.add_argument("--latent_cache_dir", type=str, default=None)
    parser.add_argument("--fid_num_images", type=int, default=None)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--use_wandb", action="store_true")
    
    args = parser.parse_args()
    
    config = TrainingConfig()
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as fp:
            cfg = yaml.safe_load(fp) or {}

        model_cfg = cfg.get("model", {})
        data_cfg = cfg.get("data", {})
        training_cfg = cfg.get("training", {})
        lora_cfg = cfg.get("lora", {})
        output_cfg = cfg.get("output", {})
        hardware_cfg = cfg.get("hardware", {})

        config.pretrained_model = model_cfg.get("pretrained_model", config.pretrained_model)
        config.data_dir = data_cfg.get("data_dir", config.data_dir)
        config.resolution = training_cfg.get("resolution", data_cfg.get("resolution", config.resolution))
        config.batch_size = training_cfg.get("batch_size", config.batch_size)
        config.num_epochs = training_cfg.get("num_epochs", config.num_epochs)
        config.learning_rate = training_cfg.get("learning_rate", config.learning_rate)
        config.lr_scheduler = training_cfg.get("lr_scheduler", config.lr_scheduler)
        config.lr_warmup_steps = training_cfg.get("lr_warmup_steps", config.lr_warmup_steps)
        config.gradient_accumulation_steps = training_cfg.get("gradient_accumulation_steps", config.gradient_accumulation_steps)
        config.max_grad_norm = training_cfg.get("max_grad_norm", config.max_grad_norm)
        config.mixed_precision = training_cfg.get("mixed_precision", config.mixed_precision)
        config.seed = training_cfg.get("seed", config.seed)
        config.lora_rank = lora_cfg.get("rank", config.lora_rank)
        config.lora_alpha = lora_cfg.get("alpha", config.lora_alpha)
        config.lora_dropout = lora_cfg.get("dropout", config.lora_dropout)
        config.lora_target_modules = lora_cfg.get("target_modules", config.lora_target_modules)
        config.output_dir = output_cfg.get("output_dir", config.output_dir)
        config.logging_dir = output_cfg.get("logging_dir", config.logging_dir)
        config.save_every_n_epochs = output_cfg.get("save_every_n_epochs", config.save_every_n_epochs)
        config.save_model_as = output_cfg.get("save_model_as", config.save_model_as)
        config.num_workers = hardware_cfg.get("num_workers", config.num_workers)
        config.use_wandb = hardware_cfg.get("use_wandb", config.use_wandb)

    if args.data_dir is not None:
        config.data_dir = args.data_dir
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.pretrained_model is not None:
        config.pretrained_model = args.pretrained_model
    if args.resolution is not None:
        config.resolution = args.resolution
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.lora_rank is not None:
        config.lora_rank = args.lora_rank
    if args.lora_alpha is not None:
        config.lora_alpha = args.lora_alpha
    if args.lora_dropout is not None:
        config.lora_dropout = args.lora_dropout
    if args.validation_split is not None:
        config.validation_split = args.validation_split
    if args.eval_every_n_epochs is not None:
        config.eval_every_n_epochs = args.eval_every_n_epochs
    if args.save_every_n_epochs is not None:
        config.save_every_n_epochs = args.save_every_n_epochs
    if args.gradient_accumulation_steps is not None:
        config.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.lr_warmup_steps is not None:
        config.lr_warmup_steps = args.lr_warmup_steps
    if args.seed is not None:
        config.seed = args.seed
    if args.log_level is not None:
        config.log_level = args.log_level
    config.use_tensorboard = not args.disable_tensorboard
    config.use_ema = not args.disable_ema
    config.enable_latent_cache = args.enable_latent_cache
    if args.latent_cache_dir is not None:
        config.latent_cache_dir = args.latent_cache_dir
    if args.fid_num_images is not None:
        config.fid_num_images = args.fid_num_images
    if args.mixed_precision is not None:
        config.mixed_precision = args.mixed_precision
    config.use_wandb = args.use_wandb
    
    # Train
    trainer = LoRATrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()