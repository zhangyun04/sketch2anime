#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to prepare paired sketch-anime dataset for training.
It processes raw images, creates sketch-image pairs, and organizes them into training and validation sets.
"""

import os
import sys
import argparse
import random
import shutil
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2


def create_sketch_from_image(image_path, output_path, method="xdog", threshold=127):
    """
    Create a sketch from an image using different methods.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the sketch
        method: Sketch extraction method ("xdog", "canny", "contour")
        threshold: Threshold for binarization
    
    Returns:
        The created sketch image
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if method == "xdog":
        # XDoG (Extended Difference of Gaussians)
        # Parameters for XDoG
        sigma = 0.5
        k = 1.6
        p = 0.98
        epsilon = 0.1
        phi = 10
        
        # Gaussian blur with sigma
        gaussian = cv2.GaussianBlur(gray, (0, 0), sigma)
        
        # Gaussian blur with sigma * k
        gaussian_k = cv2.GaussianBlur(gray, (0, 0), sigma * k)
        
        # Extended DoG
        dog = gaussian - p * gaussian_k
        
        # XDoG
        e_pow = np.exp((dog - epsilon) * phi)
        mask = (dog >= epsilon) * 1.0 + (dog < epsilon) * e_pow
        mask = np.clip(mask * 255, 0, 255).astype(np.uint8)
        
        # Threshold
        _, sketch = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    
    elif method == "canny":
        # Canny edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        sketch = cv2.Canny(blurred, threshold // 2, threshold)
        
        # Invert (edges are white on black background)
        sketch = 255 - sketch
    
    elif method == "contour":
        # Extract contours
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create sketch by drawing contours
        sketch = np.ones_like(gray) * 255
        cv2.drawContours(sketch, contours, -1, (0, 0, 0), 2)
    
    else:
        print(f"Error: Unknown sketch method '{method}'")
        return None
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the sketch
    cv2.imwrite(output_path, sketch)
    
    return sketch


def process_dataset(input_dir, output_dir, sketch_method="xdog", threshold=127, split_ratio=0.9, min_size=512):
    """
    Process the dataset and create paired sketch-image data.
    
    Args:
        input_dir: Directory containing raw anime images
        output_dir: Directory to save processed data
        sketch_method: Method to create sketches
        threshold: Threshold for sketch creation
        split_ratio: Train/validation split ratio
        min_size: Minimum image size to include
    """
    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "validation")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Get all image files
    image_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} images in {input_dir}")
    
    # Shuffle the image files
    random.shuffle(image_files)
    
    # Calculate train/validation split
    split_idx = int(len(image_files) * split_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"Train: {len(train_files)}, Validation: {len(val_files)}")
    
    # Process training images
    process_image_set(train_files, train_dir, sketch_method, threshold, min_size)
    
    # Process validation images
    process_image_set(val_files, val_dir, sketch_method, threshold, min_size)
    
    # Create metadata file
    metadata = {
        "dataset_info": {
            "total_images": len(image_files),
            "train_images": len(train_files),
            "validation_images": len(val_files),
            "sketch_method": sketch_method,
            "threshold": threshold,
            "min_size": min_size
        }
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Dataset processing complete. Metadata saved to {os.path.join(output_dir, 'metadata.json')}")


def process_image_set(image_files, output_dir, sketch_method, threshold, min_size):
    """
    Process a set of images and create sketch-image pairs.
    
    Args:
        image_files: List of image file paths
        output_dir: Directory to save processed data
        sketch_method: Method to create sketches
        threshold: Threshold for sketch creation
        min_size: Minimum image size to include
    """
    # Process each image
    skipped = 0
    for i, image_path in enumerate(tqdm(image_files, desc=f"Processing {os.path.basename(output_dir)} set")):
        try:
            # Check image size
            with Image.open(image_path) as img:
                width, height = img.size
                if width < min_size or height < min_size:
                    skipped += 1
                    continue
            
            # Create output paths
            base_name = f"image_{i:06d}"
            output_image_path = os.path.join(output_dir, f"{base_name}.png")
            output_sketch_path = os.path.join(output_dir, f"{base_name}_sketch.png")
            
            # Copy original image
            shutil.copy(image_path, output_image_path)
            
            # Create sketch
            create_sketch_from_image(
                image_path=image_path,
                output_path=output_sketch_path,
                method=sketch_method,
                threshold=threshold
            )
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            skipped += 1
    
    print(f"Processed {len(image_files) - skipped} images. Skipped {skipped} images.")


def main():
    parser = argparse.ArgumentParser(description="Prepare paired sketch-anime dataset for training")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing anime images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed data")
    parser.add_argument("--sketch_method", type=str, default="xdog", choices=["xdog", "canny", "contour"], 
                        help="Method to create sketches")
    parser.add_argument("--threshold", type=int, default=127, help="Threshold for sketch creation")
    parser.add_argument("--split_ratio", type=float, default=0.9, help="Train/validation split ratio")
    parser.add_argument("--min_size", type=int, default=512, help="Minimum image size to include")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Process the dataset
    process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sketch_method=args.sketch_method,
        threshold=args.threshold,
        split_ratio=args.split_ratio,
        min_size=args.min_size
    )


if __name__ == "__main__":
    main() 