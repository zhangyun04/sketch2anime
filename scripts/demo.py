#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple demo script to showcase the Sketch2Anime model functionality.
Upload sketches, select parameters, and generate anime images.
"""

import os
import sys
import argparse
import torch
import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.generate import load_model, generate_image
from data_processing.sketch_utils import process_sketch, apply_canny

# Default values
DEFAULT_MODEL_PATH = "./results/latest"
DEFAULT_CONFIG_PATH = "./configs/train_config.yaml"
DEFAULT_PROMPT = "anime illustration, high quality, detailed, white background"
DEFAULT_NEGATIVE_PROMPT = "low quality, ugly, deformed, malformed, text, watermark"


def setup_demo(model_path=None, config_path=None):
    """Set up the demo environment, load the model"""
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    # Check if model path exists
    if not os.path.exists(model_path) and model_path == DEFAULT_MODEL_PATH:
        print(f"Warning: Default model path {model_path} doesn't exist. Using pre-trained Canny adapter instead.")
        model_path = None
    
    # Check if config file exists
    if not os.path.exists(config_path) and config_path == DEFAULT_CONFIG_PATH:
        print(f"Warning: Default config file {config_path} doesn't exist. Using default configuration.")
        config_path = None
    
    # Load model
    try:
        pipeline, config = load_model(model_path, config_path)
        return pipeline, config
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


def sketch_to_anime(
    sketch_image, 
    prompt=DEFAULT_PROMPT, 
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    use_canny=False,
    canny_low=100,
    canny_high=200,
    guidance_scale=7.5,
    adapter_conditioning_scale=0.8,
    num_inference_steps=30,
    seed=None,
    pipeline=None,
    config=None
):
    """Generate anime image from sketch"""
    if pipeline is None:
        return None, "Model not loaded, please check error logs"
    
    # Save temporary sketch file
    temp_sketch_path = "temp_sketch.png"
    sketch_image.save(temp_sketch_path)
    
    try:
        # Set random seed
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        # Update configuration
        config["adapter_conditioning_scale"] = adapter_conditioning_scale
        config["guidance_scale"] = guidance_scale
        config["num_inference_steps"] = num_inference_steps
        
        # Set processing method
        if use_canny:
            config["training_approach"] = "canny_edge"
            config["canny_low_threshold"] = canny_low
            config["canny_high_threshold"] = canny_high
        else:
            config["training_approach"] = "direct_sketch"
        
        # Generate image
        output_image = generate_image(
            pipeline=pipeline,
            sketch_path=temp_sketch_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            config=config,
            seed=seed
        )
        
        # If using Canny, show edge image for reference
        if use_canny:
            # Process sketch to Canny edges
            processed_sketch, _ = process_sketch(
                temp_sketch_path, 
                method="canny",
                canny_low=canny_low,
                canny_high=canny_high
            )
            
            # Create comparison figure
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(sketch_image, cmap='gray')
            axs[0].set_title('Input Sketch')
            axs[0].axis('off')
            
            axs[1].imshow(processed_sketch, cmap='gray')
            axs[1].set_title('Canny Edges')
            axs[1].axis('off')
            
            axs[2].imshow(output_image)
            axs[2].set_title('Generated Result')
            axs[2].axis('off')
            
            plt.tight_layout()
            
            # Save comparison image
            comparison_path = "comparison.png"
            plt.savefig(comparison_path)
            plt.close(fig)
            
            comparison_image = Image.open(comparison_path)
            return comparison_image, "Generation successful! Left: input sketch, Middle: Canny edges, Right: generated result."
        else:
            return output_image, "Generation successful!"
            
    except Exception as e:
        return None, f"Error generating image: {e}"
    finally:
        # Clean up temporary files
        if os.path.exists(temp_sketch_path):
            os.remove(temp_sketch_path)
        if os.path.exists("comparison.png"):
            os.remove("comparison.png")


def main():
    parser = argparse.ArgumentParser(description="Sketch2Anime Demo Interface")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Model path")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Config file path")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    args = parser.parse_args()
    
    # Load model
    pipeline, config = setup_demo(args.model_path, args.config)
    
    # Set up Gradio interface
    with gr.Blocks(title="Sketch2Anime - Sketch to Anime Conversion") as demo:
        gr.Markdown("# Sketch2Anime - Sketch to Anime Image Conversion")
        gr.Markdown("Upload a sketch to generate a beautiful anime-style image")
        
        with gr.Row():
            with gr.Column():
                sketch_input = gr.Image(label="Input Sketch", type="pil", image_mode="RGB")
                
                with gr.Accordion("Basic Settings", open=True):
                    prompt = gr.Textbox(label="Prompt", value=DEFAULT_PROMPT)
                    negative_prompt = gr.Textbox(label="Negative Prompt", value=DEFAULT_NEGATIVE_PROMPT)
                    
                    with gr.Row():
                        use_canny = gr.Checkbox(label="Use Canny Edge Detection", value=False)
                        seed = gr.Number(label="Random Seed (leave empty for random)", value=None)
                
                with gr.Accordion("Advanced Settings", open=False):
                    with gr.Row():
                        canny_low = gr.Slider(label="Canny Low Threshold", minimum=50, maximum=200, value=100, step=1, visible=False)
                        canny_high = gr.Slider(label="Canny High Threshold", minimum=100, maximum=300, value=200, step=1, visible=False)
                    
                    with gr.Row():
                        guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=15.0, value=7.5, step=0.1)
                        adapter_scale = gr.Slider(label="Adapter Conditioning Scale", minimum=0.1, maximum=1.5, value=0.8, step=0.05)
                    
                    steps = gr.Slider(label="Inference Steps", minimum=10, maximum=50, value=30, step=1)
                
                generate_btn = gr.Button("Generate Image", variant="primary")
            
            with gr.Column():
                output_image = gr.Image(label="Generated Result")
                output_message = gr.Textbox(label="Message")
        
        # Set Canny parameter visibility
        use_canny.change(
            fn=lambda x: [gr.update(visible=x), gr.update(visible=x)],
            inputs=[use_canny],
            outputs=[canny_low, canny_high]
        )
        
        # Generate button click event
        generate_btn.click(
            fn=sketch_to_anime,
            inputs=[
                sketch_input, prompt, negative_prompt,
                use_canny, canny_low, canny_high,
                guidance_scale, adapter_scale, steps, seed
            ],
            outputs=[output_image, output_message],
            kwargs={"pipeline": pipeline, "config": config}
        )
        
        gr.Markdown("## Instructions")
        gr.Markdown("""
        1. Upload a sketch image, preferably with clear black lines on white background
        2. Use prompts to guide the image generation style and content
        3. Optionally choose whether to use Canny edge detection, suitable for different types of sketches
        4. Adjust advanced parameters for better results
        5. Click the Generate button to start generation
        """)
    
    # Launch Gradio interface
    demo.launch(server_port=args.port)


if __name__ == "__main__":
    main() 