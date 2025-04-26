#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for fine-tuning T2I-Adapter model to work directly with sketches.
"""

import os
import sys
import argparse
import logging
import math
import yaml
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

import numpy as np
from PIL import Image
import datasets
from huggingface_hub import HfFolder, Repository, create_repo, whoami

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLAdapterPipeline,
    T2IAdapter,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

import transformers
from transformers import AutoTokenizer, PretrainedConfig

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from tqdm.auto import tqdm


# Will error if the minimal version of diffusers is not installed
check_min_version("0.23.0")

logger = get_logger(__name__)


class SketchImageDataset(Dataset):
    """
    Dataset for T2I-Adapter training with sketch-image pairs.
    """
    def __init__(
        self,
        data_root,
        tokenizer,
        size=1024,
        center_crop=False,
        random_flip=False,
        prompt=None,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.size = size
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.default_prompt = prompt if prompt is not None else "anime illustration"

        self.sketches = []
        self.images = []
        self.prompts = []
        
        # Find all image pairs in the data directory
        for root, _, files in os.walk(data_root):
            for file in files:
                if file.endswith('_sketch.png') or file.endswith('_sketch.jpg'):
                    sketch_path = os.path.join(root, file)
                    base_name = file.replace('_sketch.png', '').replace('_sketch.jpg', '')
                    
                    # Find corresponding image file
                    img_path = None
                    for ext in ['.png', '.jpg', '.jpeg']:
                        candidate = os.path.join(root, f"{base_name}{ext}")
                        if os.path.exists(candidate):
                            img_path = candidate
                            break
                    
                    # Find corresponding prompt file (optional)
                    prompt_path = os.path.join(root, f"{base_name}.txt")
                    prompt = self.default_prompt
                    if os.path.exists(prompt_path):
                        with open(prompt_path, 'r') as f:
                            prompt = f.read().strip()
                    
                    if img_path:
                        self.sketches.append(sketch_path)
                        self.images.append(img_path)
                        self.prompts.append(prompt)
        
        self.num_samples = len(self.sketches)
        self.tokenizer = tokenizer
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Load sketch
        sketch = Image.open(self.sketches[idx]).convert("L")  # Load as grayscale
        
        # Load target image
        image = Image.open(self.images[idx]).convert("RGB")
        
        # Resize and transform
        if self.center_crop:
            sketch = self._center_crop_resize(sketch)
            image = self._center_crop_resize(image)
        else:
            sketch = sketch.resize((self.size, self.size), Image.LANCZOS)
            image = image.resize((self.size, self.size), Image.LANCZOS)
        
        # Random flip for data augmentation
        if self.random_flip and np.random.random() > 0.5:
            sketch = sketch.transpose(Image.FLIP_LEFT_RIGHT)
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Convert to tensors and normalize
        sketch_tensor = torch.from_numpy(np.array(sketch)).float() / 255.0
        sketch_tensor = sketch_tensor.unsqueeze(0)  # Add channel dimension [1, H, W]
        
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        
        # Get prompt
        prompt = self.prompts[idx]
        
        return {
            "sketch": sketch_tensor,
            "image": image_tensor,
            "prompt": prompt,
        }
    
    def _center_crop_resize(self, image):
        width, height = image.size
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        right = left + min_dim
        bottom = top + min_dim
        
        image = image.crop((left, top, right, bottom))
        image = image.resize((self.size, self.size), Image.LANCZOS)
        return image


def load_config(config_path):
    """Load configuration from yaml file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune T2I-Adapter for sketch-to-anime conversion")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/train_config.yaml",
        help="Path to the configuration file"
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model to the Hugging Face Hub",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local model"
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="Hugging Face token to use for pushing to the Hub"
    )
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Initialize accelerator
    logging_dir = Path(config["logging_dir"])
    accelerator_project_config = ProjectConfiguration(
        project_dir=config["training"]["output_dir"],
        logging_dir=logging_dir
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        mixed_precision=config["mixed_precision"],
        project_config=accelerator_project_config,
    )
    
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    # Set seed for reproducibility
    if config["training"]["seed"] is not None:
        set_seed(config["training"]["seed"])
    
    # Create directories
    os.makedirs(config["training"]["output_dir"], exist_ok=True)
    os.makedirs(config["logging_dir"], exist_ok=True)
    
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = os.path.basename(config["training"]["output_dir"])
                repo_owner = whoami()["name"]
                hub_model_id = f"{repo_owner}/{repo_name}"
            else:
                hub_model_id = args.hub_model_id
            
            repo = create_repo(
                hub_model_id,
                exist_ok=True,
                token=args.hub_token,
            )
            repo.git_pull()
            
            repo = Repository(
                config["training"]["output_dir"],
                clone_from=hub_model_id,
                token=args.hub_token,
            )
    
    # Load models
    logger.info("Loading models...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["pretrained_model_name_or_path"],
        subfolder="tokenizer",
        revision=config["model"]["revision"],
    )
    
    # Load adapter model
    adapter = T2IAdapter.from_pretrained(
        config["model"]["adapter_model_name_or_path"],
        torch_dtype=torch.float16 if config["mixed_precision"] == "fp16" else torch.float32,
        variant=config["model"]["variant"],
    )
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        config["model"]["vae_model_name_or_path"],
        torch_dtype=torch.float16 if config["mixed_precision"] == "fp16" else torch.float32,
    )
    
    # Load noise scheduler for training
    noise_scheduler = DDPMScheduler.from_pretrained(
        config["model"]["pretrained_model_name_or_path"],
        subfolder="scheduler",
    )
    
    # Load UNet
    unet = UNet2DConditionModel.from_pretrained(
        config["model"]["pretrained_model_name_or_path"],
        subfolder="unet",
        torch_dtype=torch.float16 if config["mixed_precision"] == "fp16" else torch.float32,
    )
    
    # Enable memory efficient attention if requested
    if config["enable_xformers_memory_efficient_attention"] and is_xformers_available():
        import xformers
        
        unet.enable_xformers_memory_efficient_attention()
        adapter.enable_xformers_memory_efficient_attention()
    
    # Enable gradient checkpointing if requested
    if config["gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()
        adapter.enable_gradient_checkpointing()
    
    # Freeze the unet and vae
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    
    # Only train the adapter
    adapter.train()
    
    # Create training dataset
    train_dataset = SketchImageDataset(
        data_root=config["dataset"]["train_data_dir"],
        tokenizer=tokenizer,
        size=config["training"]["resolution"],
        center_crop=config["dataset"]["center_crop"],
        random_flip=config["dataset"]["random_flip"],
    )
    
    # DataLoaders creation
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["training"]["train_batch_size"],
        shuffle=True,
        num_workers=4,
    )
    
    # Create optimizer
    params_to_optimize = adapter.parameters()
    
    if config["use_8bit_adam"]:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install bitsandbytes using: pip install bitsandbytes"
            )
    else:
        optimizer_cls = torch.optim.AdamW
    
    optimizer = optimizer_cls(
        params_to_optimize,
        lr=config["training"]["learning_rate"],
        betas=(config["optimizer"]["adam_beta1"], config["optimizer"]["adam_beta2"]),
        weight_decay=config["optimizer"]["adam_weight_decay"],
        eps=config["optimizer"]["adam_epsilon"],
    )
    
    # Scheduler
    lr_scheduler = get_scheduler(
        config["training"]["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=config["training"]["lr_warmup_steps"],
        num_training_steps=config["training"]["max_train_steps"],
    )
    
    # Prepare everything with accelerator
    adapter, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        adapter, optimizer, train_dataloader, lr_scheduler
    )
    
    # Calculate number of steps
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config["training"]["gradient_accumulation_steps"]
    )
    
    max_train_steps = config["training"]["max_train_steps"]
    if max_train_steps is None:
        max_train_steps = config["training"]["num_train_epochs"] * num_update_steps_per_epoch
    
    total_batch_size = (
        config["training"]["train_batch_size"] 
        * accelerator.num_processes 
        * config["training"]["gradient_accumulation_steps"]
    )
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {config['training']['train_batch_size']}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config['training']['gradient_accumulation_steps']}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    
    global_step = 0
    first_epoch = 0
    
    # Training loop
    progress_bar = tqdm(
        range(global_step, max_train_steps),
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    
    for epoch in range(first_epoch, config["training"]["num_train_epochs"]):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(adapter):
                # Get the inputs
                sketch = batch["sketch"].to(accelerator.device)
                target_images = batch["image"].to(accelerator.device)
                
                # Encode target images
                vae.to(accelerator.device)
                latents = vae.encode(target_images).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Sample noise for the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, 
                    noise_scheduler.config.num_train_timesteps, 
                    (bsz,), 
                    device=latents.device
                )
                timesteps = timesteps.long()
                
                # Add noise to the latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Use sketch directly as adapter conditioning
                adapter_condition = sketch
                
                # Get adapter features from the sketch
                down_block_res_samples, mid_block_res_sample = adapter(adapter_condition)
                
                # Predict the noise residual using UNet with adapter conditioning
                unet.to(accelerator.device)
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=None,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample
                
                # Calculate loss
                loss = F.mse_loss(model_pred, noise, reduction="mean")
                
                # Backpropagate
                accelerator.backward(loss)
                
                # Clip gradients
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(adapter.parameters(), 1.0)
                
                # Update parameters
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update progress bar
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # Save model checkpoint
                if global_step % config["training"]["checkpointing_steps"] == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(
                            config["training"]["output_dir"], f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        
                        # Create pipeline for validation
                        pipeline = StableDiffusionXLAdapterPipeline.from_pretrained(
                            config["model"]["pretrained_model_name_or_path"],
                            vae=vae,
                            adapter=accelerator.unwrap_model(adapter),
                            torch_dtype=torch.float16 if config["mixed_precision"] == "fp16" else torch.float32,
                        )
                        
                        # Save adapter model
                        adapter_state_dict = accelerator.unwrap_model(adapter).state_dict()
                        torch.save(
                            adapter_state_dict,
                            os.path.join(save_path, "adapter_model.safetensors")
                        )
                        
                        # Run validation
                        if (
                            global_step % config["training"]["validation_steps"] == 0
                            and config["training"]["validation_prompt"] is not None
                        ):
                            logger.info(f"Running validation... \n Generating image with prompt: {config['training']['validation_prompt']}")
                            
                            # Select random sketch from validation dataset
                            val_dataset = SketchImageDataset(
                                data_root=config["dataset"]["validation_data_dir"],
                                tokenizer=tokenizer,
                                size=config["training"]["resolution"],
                                center_crop=True,
                            )
                            
                            if len(val_dataset) > 0:
                                val_sample = val_dataset[np.random.randint(0, len(val_dataset))]
                                val_sketch = val_sample["sketch"].unsqueeze(0).to(accelerator.device)
                                
                                # Generate image
                                pipeline.to(accelerator.device)
                                image = pipeline(
                                    prompt=config["training"]["validation_prompt"],
                                    negative_prompt=config["training"]["negative_prompt"],
                                    image=val_sketch,
                                    num_inference_steps=config["num_inference_steps"],
                                    adapter_conditioning_scale=config["adapter_conditioning_scale"],
                                    guidance_scale=config["guidance_scale"],
                                ).images[0]
                                
                                # Save validation image
                                image.save(os.path.join(save_path, f"validation_{global_step}.png"))
                        
                        # Push to hub
                        if args.push_to_hub and accelerator.is_main_process:
                            repo.push_to_hub(
                                commit_message=f"Checkpoint {global_step}",
                                blocking=False,
                                auto_lfs_prune=True,
                            )
            
            # Check if we've reached max steps
            if global_step >= max_train_steps:
                break
    
    # Final save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        adapter = accelerator.unwrap_model(adapter)
        
        # Save the adapter
        adapter.save_pretrained(os.path.join(config["training"]["output_dir"], "adapter_model_final"))
        
        # Push to hub
        if args.push_to_hub:
            repo.push_to_hub(commit_message="Final model", blocking=False, auto_lfs_prune=True)
    
    accelerator.end_training()


if __name__ == "__main__":
    main() 