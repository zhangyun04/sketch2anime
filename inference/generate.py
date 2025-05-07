#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for generating anime-style images from sketches using the fine-tuned T2I-Adapter model.
"""

import os
import sys
import argparse
import logging
import yaml
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    EulerAncestralDiscreteScheduler,
    StableDiffusionXLAdapterPipeline,
    T2IAdapter,
)
from controlnet_aux.canny import CannyDetector
from diffusers.utils import load_image, make_image_grid

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.sketch_utils import process_sketch, find_optimal_canny_params


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_model(model_path=None, config_path=None):
    """
    Load the T2I-Adapter pipeline for inference.
    
    Args:
        model_path: Path to the fine-tuned adapter model
        config_path: Path to the configuration file
    
    Returns:
        The loaded pipeline
    """
    # Use default configuration if not provided
    if config_path:
        config = load_config(config_path)
    else:
        # Default configuration
        config = {
            "model": {
                "pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0",
                "vae_model_name_or_path": "madebyollin/sdxl-vae-fp16-fix",
                "variant": "fp16",
                "inference_scheduler": "euler_ancestral"
            },
            "adapter_conditioning_scale": 0.8,
            "guidance_scale": 7.5,
            "num_inference_steps": 30,
            "mixed_precision": "fp16"
        }
    
    # Load the VAE
    vae = AutoencoderKL.from_pretrained(
        config["model"]["vae_model_name_or_path"],
        torch_dtype=torch.float16 if config.get("mixed_precision") == "fp16" else torch.float32,
    )
    
    # Load the scheduler based on configuration
    inference_scheduler_type = config["model"].get("inference_scheduler", "euler_ancestral")
    if inference_scheduler_type.lower() == "ddpm":
        scheduler = DDPMScheduler.from_pretrained(
            config["model"]["pretrained_model_name_or_path"],
            subfolder="scheduler"
        )
        print("Using DDPMScheduler for inference")
    else:
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
            config["model"]["pretrained_model_name_or_path"],
            subfolder="scheduler"
        )
        print("Using EulerAncestralDiscreteScheduler for inference")
    
    # Load the adapter model
    if model_path:
        adapter = T2IAdapter.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if config.get("mixed_precision") == "fp16" else torch.float32,
            variant=config["model"].get("variant")
        )
    else:
        # Use the default canny adapter if no fine-tuned model is provided
        adapter = T2IAdapter.from_pretrained(
            "TencentARC/t2i-adapter-canny-sdxl-1.0",
            torch_dtype=torch.float16 if config.get("mixed_precision") == "fp16" else torch.float32,
            variant=config["model"].get("variant")
        )
    
    # Load the pipeline
    pipeline = StableDiffusionXLAdapterPipeline.from_pretrained(
        config["model"]["pretrained_model_name_or_path"],
        vae=vae,
        adapter=adapter,
        scheduler=scheduler,
        torch_dtype=torch.float16 if config.get("mixed_precision") == "fp16" else torch.float32,
        variant=config["model"].get("variant"),
    )
    
    # Enable xformers for memory efficiency if available
    if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
        pipeline.enable_xformers_memory_efficient_attention()
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = pipeline.to(device)
    
    return pipeline, config


def generate_image(pipeline, sketch_path, prompt=None, negative_prompt=None, config=None, seed=None):
    """
    Generate an image from a sketch using the T2I-Adapter pipeline.
    
    Args:
        pipeline: The loaded pipeline
        sketch_path: Path to the sketch image
        prompt: Text prompt for generation
        negative_prompt: Negative text prompt
        config: Pipeline configuration
        seed: Random seed for reproducibility
        
    Returns:
        Generated image
    """
    if config is None:
        config = {}
    
    # Set default parameters
    adapter_conditioning_scale = config.get("adapter_conditioning_scale", 0.8)
    guidance_scale = config.get("guidance_scale", 7.5)
    num_inference_steps = config.get("num_inference_steps", 30)
    
    # Set default prompts if not provided
    if prompt is None:
        prompt = "anime style, high quality, detailed, vivid colors, white background"
    if negative_prompt is None:
        negative_prompt = "graphic, text, watermark, low quality, ugly, deformed, malformed"
    
    # Set seed for reproducibility if provided
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    # Determine the processing method from config
    processing_method = config.get("training_approach", "direct_sketch")
    
    # Process the sketch using the utility function
    processed_image, conditioning_tensor = process_sketch(
        sketch_path, 
        method="canny" if processing_method == "canny_edge" else "direct",
        canny_low=config.get("canny_low_threshold", 100),
        canny_high=config.get("canny_high_threshold", 200),
        target_size=config.get("training", {}).get("resolution", 1024)
    )
    
    # Move tensor to the same device as the pipeline
    conditioning_tensor = conditioning_tensor.to(pipeline.device)
    
    # Generate the image
    image = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=conditioning_tensor,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        adapter_conditioning_scale=adapter_conditioning_scale,
    ).images[0]
    
    return image


def process_sketch(sketch_path, output_path, pipeline, config, prompt=None, negative_prompt=None, seed=None):
    """
    Process a sketch image and generate an anime-style image.
    
    Args:
        sketch_path: Path to the sketch image
        output_path: Path to save the generated image
        pipeline: The StableDiffusionXLAdapterPipeline
        config: Configuration dictionary
        prompt: Text prompt to guide the generation
        negative_prompt: Negative text prompt
        seed: Random seed for reproducibility
    
    Returns:
        The generated image
    """
    # Set default prompt if not provided
    if prompt is None:
        prompt = "anime style, high quality, same pose, manga illustration, smooth and simple line, vivid, white background"
    
    # Set default negative prompt if not provided
    if negative_prompt is None:
        negative_prompt = "graphic, text, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured, extra digit, fewer digits, cropped, worst quality, low quality"
    
    # Set random seed for reproducibility if provided
    if seed is not None:
        generator = torch.Generator(pipeline.device).manual_seed(seed)
    else:
        generator = None
    
    # Load and preprocess the sketch
    sketch = Image.open(sketch_path).convert("L")  # Convert to grayscale
    
    # Generate the image
    output = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=sketch,
        num_inference_steps=config.get("num_inference_steps", 30),
        adapter_conditioning_scale=config.get("adapter_conditioning_scale", 0.8),
        guidance_scale=config.get("guidance_scale", 7.5),
        generator=generator,
    )
    
    # Save the generated image
    output_image = output.images[0]
    output_image.save(output_path)
    
    # Create a comparison visualization
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(sketch, cmap='gray')
    plt.title("Input Sketch")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(np.array(output_image))
    plt.title("Generated Image")
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save comparison
    comparison_path = os.path.splitext(output_path)[0] + "_comparison.png"
    plt.savefig(comparison_path)
    plt.close()
    
    print(f"Generated image saved to: {output_path}")
    print(f"Comparison image saved to: {comparison_path}")
    
    return output_image


def batch_process(input_dir, output_dir, pipeline, config, prompt=None, negative_prompt=None, seed=None):
    """
    Process all sketch images in a directory.
    
    Args:
        input_dir: Directory containing sketch images
        output_dir: Directory to save generated images
        pipeline: The StableDiffusionXLAdapterPipeline
        config: Configuration dictionary
        prompt: Text prompt to guide the generation
        negative_prompt: Negative text prompt
        seed: Random seed for reproducibility
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all sketch images in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            sketch_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{base_name}_generated.png")
            
            print(f"Processing {filename}...")
            process_sketch(
                sketch_path=sketch_path,
                output_path=output_path,
                pipeline=pipeline,
                config=config,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed
            )


def main():
    parser = argparse.ArgumentParser(description="Generate anime images from sketches using T2I-Adapter")
    parser.add_argument("--sketch", type=str, help="Path to the input sketch image")
    parser.add_argument("--output", type=str, help="Path to save the generated image")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the fine-tuned adapter model")
    parser.add_argument("--config", type=str, default=None, help="Path to the configuration file")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt to guide the generation")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative prompt to guide the generation")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--batch", action="store_true", help="Process all images in a directory")
    parser.add_argument("--input_dir", type=str, help="Directory containing sketch images (for batch processing)")
    parser.add_argument("--output_dir", type=str, help="Directory to save generated images (for batch processing)")
    
    args = parser.parse_args()
    
    # Load the model and configuration
    pipeline, config = load_model(args.model_path, args.config)
    
    # Batch processing mode
    if args.batch:
        if not args.input_dir or not args.output_dir:
            raise ValueError("For batch processing, both --input_dir and --output_dir must be specified")
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Process all images in the input directory
        for filename in os.listdir(args.input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                sketch_path = os.path.join(args.input_dir, filename)
                output_path = os.path.join(args.output_dir, f"generated_{filename}")
                
                print(f"Processing {filename}...")
                try:
                    # Generate the image
                    image = generate_image(
                        pipeline=pipeline,
                        sketch_path=sketch_path,
                        prompt=args.prompt,
                        negative_prompt=args.negative_prompt,
                        config=config,
                        seed=args.seed
                    )
                    
                    # Save the image
                    image.save(output_path)
                    print(f"Saved to {output_path}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
    
    # Single image processing mode
    else:
        if not args.sketch or not args.output:
            raise ValueError("Both --sketch and --output must be specified")
        
        # Generate the image
        image = generate_image(
            pipeline=pipeline,
            sketch_path=args.sketch,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            config=config,
            seed=args.seed
        )
        
        # Save the image
        image.save(args.output)
        print(f"Generated image saved to {args.output}")


if __name__ == "__main__":
    main() 