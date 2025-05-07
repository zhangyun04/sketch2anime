#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilities for sketch processing and edge detection.
Includes functions from the t2imodel_v2 notebook.
"""

import os
import numpy as np
import cv2
from PIL import Image, ImageOps
import torch
from typing import Tuple, Optional, Union


def resize_image(image, target_size=1024):
    """
    Resize an image to the target size while preserving aspect ratio.
    
    Args:
        image: PIL Image to resize
        target_size: Target size (width and height)
        
    Returns:
        Resized PIL Image
    """
    width, height = image.size
    scaling_factor = target_size / max(width, height)
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Create a white background image of target size
    background = Image.new('RGB', (target_size, target_size), (255, 255, 255))
    
    # Paste the resized image in the center
    offset_x = (target_size - new_width) // 2
    offset_y = (target_size - new_height) // 2
    background.paste(resized_image, (offset_x, offset_y))
    
    return background


def apply_canny(image, low_threshold=100, high_threshold=200):
    """
    Apply Canny edge detection to an image.
    
    Args:
        image: PIL Image or numpy array
        low_threshold: Lower threshold for Canny edge detection
        high_threshold: Higher threshold for Canny edge detection
        
    Returns:
        PIL Image with Canny edges
    """
    if isinstance(image, Image.Image):
        # Convert PIL image to numpy array
        image_np = np.array(image.convert("L"))
    else:
        # Assume it's already a numpy array
        image_np = image
        if len(image_np.shape) == 3:
            # Convert RGB to grayscale
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(image_np, low_threshold, high_threshold)
    
    # Convert back to PIL
    return Image.fromarray(edges)


def process_sketch(sketch_path, method="direct", canny_low=100, canny_high=200, target_size=1024):
    """
    Process a sketch image for use as adapter conditioning.
    
    Args:
        sketch_path: Path to the sketch image
        method: Processing method - "direct" or "canny"
        canny_low: Lower threshold for Canny edge detection
        canny_high: Higher threshold for Canny edge detection
        target_size: Target size for the output image
        
    Returns:
        Processed PIL Image and tensor ready for model input
    """
    # Load the sketch
    sketch = Image.open(sketch_path).convert("L")
    
    # Resize the sketch
    sketch = resize_image(sketch, target_size)
    
    if method.lower() == "canny":
        # Apply Canny edge detection
        processed = apply_canny(sketch, canny_low, canny_high)
    else:
        # Use the sketch directly
        processed = sketch
    
    # Convert to tensor and normalize
    tensor = torch.from_numpy(np.array(processed)).float() / 255.0
    tensor = tensor.unsqueeze(0)  # Add channel dimension [1, H, W]
    
    return processed, tensor


def find_optimal_canny_params(image_path, min_low=50, max_low=150, min_high=150, max_high=250, step=10):
    """
    Find optimal Canny parameters for a given image by grid search.
    
    Args:
        image_path: Path to the image
        min_low: Minimum value for low threshold
        max_low: Maximum value for low threshold
        min_high: Minimum value for high threshold
        max_high: Maximum value for high threshold
        step: Step size for parameter search
        
    Returns:
        Dictionary of optimal parameters and results
    """
    image = Image.open(image_path).convert("L")
    image_np = np.array(image)
    
    best_score = -float('inf')
    best_params = None
    best_edges = None
    
    for low in range(min_low, max_low + 1, step):
        for high in range(low + step, max_high + 1, step):
            # Apply Canny
            edges = cv2.Canny(image_np, low, high)
            
            # Calculate score based on edge density and continuity
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Simple continuity metric using dilation and edge count ratio
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)
            continuity = np.sum(edges > 0) / (np.sum(dilated > 0) + 1e-6)
            
            # Combined score - we want reasonable density with good continuity
            score = (edge_density * 0.4) + (continuity * 0.6)
            
            if score > best_score:
                best_score = score
                best_params = (low, high)
                best_edges = edges
    
    return {
        "optimal_low": best_params[0],
        "optimal_high": best_params[1],
        "edge_density": edge_density,
        "continuity": continuity,
        "score": best_score,
        "edges": Image.fromarray(best_edges)
    }


def create_batch_tensors(image_paths, method="direct", canny_low=100, canny_high=200, target_size=1024):
    """
    Create a batch of tensors from multiple images.
    
    Args:
        image_paths: List of paths to images
        method: Processing method - "direct" or "canny"
        canny_low: Lower threshold for Canny edge detection
        canny_high: Higher threshold for Canny edge detection
        target_size: Target size for the output images
        
    Returns:
        Batch tensor ready for model input [B, 1, H, W]
    """
    tensors = []
    
    for path in image_paths:
        _, tensor = process_sketch(path, method, canny_low, canny_high, target_size)
        tensors.append(tensor)
    
    return torch.cat(tensors, dim=0) 