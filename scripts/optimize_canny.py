#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to optimize Canny edge detection parameters for sketches.
This helps minimize information loss when converting sketches to edge maps.
"""

import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from itertools import product


def calculate_similarity(sketch, edge):
    """
    Calculate structural similarity between sketch and generated edge map.
    Higher score means better preservation of sketch details.
    
    Args:
        sketch: Original sketch image (grayscale)
        edge: Generated edge image (grayscale)
    
    Returns:
        float: Similarity score
    """
    # Normalize images
    sketch = sketch.astype(float) / 255.0
    edge = edge.astype(float) / 255.0
    
    # Calculate structural similarity
    # Simple pixel-wise similarity (can be replaced with SSIM)
    diff = np.abs(sketch - edge)
    similarity = 1.0 - np.mean(diff)
    
    return similarity


def process_image(args):
    """Process a single image with given Canny parameters"""
    image_path, low_threshold, high_threshold, gaussian_blur = args
    
    # Read image and convert to grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # Apply Gaussian blur
    if gaussian_blur > 0:
        blurred = cv2.GaussianBlur(img, (gaussian_blur, gaussian_blur), 0)
    else:
        blurred = img
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
    # Calculate similarity score
    score = calculate_similarity(img, edges)
    
    return {
        'low_threshold': low_threshold,
        'high_threshold': high_threshold,
        'gaussian_blur': gaussian_blur,
        'score': score,
        'image_path': image_path
    }


def optimize_parameters(input_dir, output_dir, num_samples=5, parallel=True):
    """
    Find optimal Canny parameters for a set of sketch images.
    
    Args:
        input_dir: Directory containing sketch images
        output_dir: Directory to save processed edges and results
        num_samples: Number of random samples to use for optimization
        parallel: Whether to use parallel processing
    
    Returns:
        dict: Optimal parameters
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    
    # Sample images if there are too many
    if len(image_files) > num_samples:
        image_files = np.random.choice(image_files, num_samples, replace=False)
    
    print(f"Optimizing parameters using {len(image_files)} images...")
    
    # Parameter ranges to test
    low_thresholds = [10, 20, 30, 50, 70, 100]
    high_thresholds = [30, 50, 100, 150, 200, 250]
    gaussian_blurs = [0, 3, 5, 7]  # 0 means no blur
    
    # Generate all parameter combinations
    param_combinations = []
    for img_path in image_files:
        for lt, ht, gb in product(low_thresholds, high_thresholds, gaussian_blurs):
            if lt < ht:  # Ensure low threshold is less than high threshold
                param_combinations.append((img_path, lt, ht, gb))
    
    # Process images
    results = []
    if parallel:
        with ProcessPoolExecutor() as executor:
            for result in tqdm(executor.map(process_image, param_combinations), 
                              total=len(param_combinations)):
                if result is not None:
                    results.append(result)
    else:
        for params in tqdm(param_combinations):
            result = process_image(params)
            if result is not None:
                results.append(result)
    
    # Group results by parameter combination
    grouped_results = {}
    for r in results:
        key = (r['low_threshold'], r['high_threshold'], r['gaussian_blur'])
        if key not in grouped_results:
            grouped_results[key] = []
        grouped_results[key].append(r['score'])
    
    # Calculate average score for each parameter combination
    avg_scores = {k: np.mean(v) for k, v in grouped_results.items()}
    
    # Find best parameters
    best_params = max(avg_scores.items(), key=lambda x: x[1])
    best_low, best_high, best_blur = best_params[0]
    best_score = best_params[1]
    
    print(f"\nBest parameters:")
    print(f"Low threshold: {best_low}")
    print(f"High threshold: {best_high}")
    print(f"Gaussian blur kernel size: {best_blur}")
    print(f"Average similarity score: {best_score:.4f}")
    
    # Save best parameters
    with open(os.path.join(output_dir, 'optimal_canny_params.txt'), 'w') as f:
        f.write(f"Low threshold: {best_low}\n")
        f.write(f"High threshold: {best_high}\n")
        f.write(f"Gaussian blur kernel size: {best_blur}\n")
        f.write(f"Average similarity score: {best_score:.4f}\n")
    
    # Generate and save edge images with optimal parameters
    for img_path in tqdm(image_files, desc="Generating optimal edge maps"):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        # Apply optimal parameters
        if best_blur > 0:
            blurred = cv2.GaussianBlur(img, (best_blur, best_blur), 0)
        else:
            blurred = img
            
        edges = cv2.Canny(blurred, best_low, best_high)
        
        # Save result
        filename = os.path.basename(img_path)
        base, _ = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{base}_edge.png")
        cv2.imwrite(output_path, edges)
        
        # Create comparison visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(img, cmap='gray')
        ax1.set_title('Original Sketch')
        ax1.axis('off')
        
        ax2.imshow(edges, cmap='gray')
        ax2.set_title('Optimized Canny Edge')
        ax2.axis('off')
        
        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base}_comparison.png"))
        plt.close(fig)
    
    return {
        'low_threshold': best_low,
        'high_threshold': best_high,
        'gaussian_blur': best_blur,
        'score': best_score
    }


def main():
    parser = argparse.ArgumentParser(description='Optimize Canny parameters for sketch-to-edge conversion')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing sketch images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save processed edges and results')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to use for optimization')
    parser.add_argument('--no_parallel', action='store_true', help='Disable parallel processing')
    
    args = parser.parse_args()
    
    optimize_parameters(
        args.input_dir,
        args.output_dir,
        args.num_samples,
        not args.no_parallel
    )


if __name__ == "__main__":
    main() 