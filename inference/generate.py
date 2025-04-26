#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for generating anime-style images from sketches using the fine-tuned T2I-Adapter model.
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from diffusers import StableDiffusionXLAdapterPipeline, EulerAncestralDiscreteScheduler, AutoencoderKL, T2IAdapter


def load_config(config_path):
    """Load configuration from yaml file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


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
                "variant": "fp16"
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
    
    # Load the scheduler
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
        config["model"]["pretrained_model_name_or_path"],
        subfolder="scheduler"
    )
    
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
    
    # Create the pipeline
    pipeline = StableDiffusionXLAdapterPipeline.from_pretrained(
        config["model"]["pretrained_model_name_or_path"],
        vae=vae,
        adapter=adapter,
        scheduler=scheduler,
        torch_dtype=torch.float16 if config.get("mixed_precision") == "fp16" else torch.float32,
        variant=config["model"].get("variant"),
    )
    
    # Enable memory efficient attention if available
    try:
        import xformers
        pipeline.enable_xformers_memory_efficient_attention()
    except ImportError:
        print("xformers not available, using default attention mechanism")
    
    return pipeline, config


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
    
    # Move the pipeline to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = pipeline.to(device)
    
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
    
    # Process images
    if args.batch:
        if not args.input_dir or not args.output_dir:
            parser.error("--input_dir and --output_dir are required for batch processing")
            
        batch_process(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            pipeline=pipeline,
            config=config,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            seed=args.seed
        )
    else:
        if not args.sketch or not args.output:
            parser.error("--sketch and --output are required for single image processing")
            
        process_sketch(
            sketch_path=args.sketch,
            output_path=args.output,
            pipeline=pipeline,
            config=config,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            seed=args.seed
        )


if __name__ == "__main__":
    main() 