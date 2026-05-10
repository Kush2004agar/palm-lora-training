# Palm LoRA Training Pipeline

Complete pipeline to train LoRA on 6,000 palm images and generate 54,000 more.

## Quick Start

### 1. Installation

```bash
# Make setup script executable
chmod +x setup.sh

# Run setup
./setup.sh

# Activate environment
source lora_env/bin/activate
```

### 2. Prepare Your Dataset

Place your 6,000 palm images in a folder, then run:

```bash
python prepare_dataset.py \
    --source "./your_6000_palms" \
    --output "./palm_dataset" \
    --resolution 512
```

### 3. Train LoRA

```bash
python train_lora.py \
    --data_dir "./palm_dataset/10_palm" \
    --output_dir "./output" \
    --num_epochs 15 \
    --batch_size 4
```

Training time: ~4-8 hours on RTX 3090

### 4. Generate New Images

```bash
python generate_images.py \
    --lora_path "./output/checkpoint-final" \
    --num_images 54000 \
    --diverse \
    --output_dir "./generated_palms"
```

## Or Run Everything At Once

```bash
chmod +x run_training.sh
./run_training.sh
```

## File Structure

```text
.
├── prepare_dataset.py      # Image preprocessing + caption generation
├── train_lora.py           # LoRA fine-tuning (with validation/FID/EMA)
├── generate_images.py      # Image generation from trained LoRA
├── filter_palms.py         # Optional CSV-based palm filtering utility
├── validate_setup.py       # Environment validation
├── config.yml              # Optional YAML config for training
├── run_training.sh         # Linux/macOS pipeline script
├── run_pipeline.ps1        # Windows pipeline script
└── requirements.txt
```

## Configuration

Training can be configured in two ways:

1. CLI arguments (highest priority)
2. YAML config via `--config config.yml`

Example:

```bash
python train_lora.py --config config.yml --num_epochs 25 --batch_size 2
```

## Improvements Included

- Data augmentation in training dataset pipeline
- Validation split and validation loss tracking
- Optional latent caching for faster training
- EMA tracking for UNet LoRA weights
- FID evaluation support (requires `torch-fidelity`)
- TensorBoard logging (`logs/`)
- Robust checkpointing with optimizer + scheduler state
- Safer, non-overwriting diverse generation flow

## TensorBoard

```bash
tensorboard --logdir ./logs
```

## Notes

- For small datasets (3k-6k images), keep captions descriptive for better LoRA quality.
- `generate_images.py --diverse` now writes unique filenames across prompts.
- `filter_palms.py` now uses CLI args instead of hardcoded local paths.