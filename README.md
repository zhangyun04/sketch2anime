# Sketch2Anime

A deep learning project that converts sketches into high-quality anime-style images.

## Project Overview

This project is based on Stable Diffusion XL and T2I-Adapter technology, enabling the generation of beautiful anime-style images from simple sketches. We provide two methods to optimize the sketch-to-image conversion process:

1. **Canny Parameter Optimization**: Optimize Canny edge detection hyperparameters to preserve as much detail as possible when converting sketches to Canny edges
2. **Model Fine-tuning**: Directly fine-tune the T2I-Adapter model using sketches as control signals

## Project Structure

```
sketch2anime/
├── configs/             # Configuration files
├── data/                # Dataset storage
├── data_processing/     # Data preprocessing code
├── inference/           # Inference code
├── models/              # Pre-trained model storage
├── scripts/             # Utility scripts
└── train/               # Training code
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sketch2anime.git
cd sketch2anime

# Install dependencies
pip install -r requirements.txt
```

## Dataset Requirements

To successfully train the model, you need to prepare a dataset with the following characteristics:

1. **Format**: Paired sketch-image data
   - Sketch file naming format: `image_name_sketch.png/jpg`
   - Corresponding image: `image_name.png/jpg/jpeg`
   - Optional text prompt: `image_name.txt` (containing image description)

2. **Quantity**: Recommended at least 500-1000 high-quality sketch-image pairs

3. **Quality Requirements**:
   - Clear sketch lines
   - High-quality anime-style images
   - Recommended resolution of 1024×1024 or scalable to this resolution

4. **Diversity**:
   - Different characters, scenes, and art styles
   - Various line styles (from simple to complex)

5. **Preprocessing**:
   - White background with black lines recommended for sketches
   - Remove background interference
   - Properly align sketch and image pairs

## Usage

### 1. Data Preparation

Place the sketch-image paired dataset in the appropriate subdirectory of the `data/` directory:

```bash
mkdir -p data/train data/validation
# Put training data in data/train
# Put validation data in data/validation
```

### 2. Canny Parameter Optimization (Optional)

Run the Canny parameter optimization experiment to find the best edge detection parameters:

```bash
python scripts/optimize_canny.py --input_dir data/sketches --output_dir data/processed
```

### 3. Model Training

There are two ways to train the model: directly using sketches or through Canny edges. You can set this in the configuration file:

```bash
# Train using config file
python train/train_adapter.py --config configs/train_config.yaml

# Or specify a model ID to upload to Hugging Face
python train/train_adapter.py --config configs/train_config.yaml --push_to_hub --hub_model_id "your-username/sketch2anime"
```

#### Training Configuration

In `configs/train_config.yaml`, you can set:

- `model.train_scheduler`: Scheduler used for training ("ddpm" or "euler_ancestral")
- `model.inference_scheduler`: Scheduler used for inference (recommended "euler_ancestral")
- `training_approach`: Training method ("direct_sketch" or "canny_edge")

### 4. Inference Generation

Use the trained model to generate images:

```bash
# Single image generation
python inference/generate.py --sketch path/to/your/sketch.jpg --output output.png --model_path ./results/checkpoint-10000

# Batch processing
python inference/generate.py --batch --input_dir ./sketches --output_dir ./outputs --model_path ./results/checkpoint-10000

# Add text prompt (optional)
python inference/generate.py --sketch path/to/your/sketch.jpg --output output.png --prompt "anime girl with blue hair"
```

## Scheduler Options

This project supports two schedulers:

1. **DDPMScheduler**:
   - More stable training process
   - Requires more sampling steps
   - Suitable for the training phase

2. **EulerAncestralDiscreteScheduler**:
   - Faster inference speed
   - Uses fewer sampling steps (20-30 steps) to get high-quality results
   - Generated results are more diverse with richer details
   - Recommended for the inference phase

## License

MIT License 