#!/bin/bash

echo "🚀 Starting Palm LoRA Training Pipeline"
echo "========================================"

# Activate environment
source lora_env/bin/activate

# Step 1: Prepare dataset
echo ""
echo "📂 Step 1: Preparing dataset..."
python prepare_dataset.py \
    --source "./your_6000_palms" \
    --output "./palm_dataset" \
    --resolution 512 \
    --caption-style simple

# Step 2: Train LoRA
echo ""
echo "🎯 Step 2: Training LoRA..."
python train_lora.py \
    --data_dir "./palm_dataset/10_palm" \
    --output_dir "./output" \
    --pretrained_model "runwayml/stable-diffusion-v1-5" \
    --resolution 512 \
    --batch_size 4 \
    --num_epochs 15 \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --mixed_precision fp16

# Step 3: Generate images
echo ""
echo "🎨 Step 3: Generating 54,000 new palm images..."
python generate_images.py \
    --lora_path "./output/checkpoint-final" \
    --base_model "runwayml/stable-diffusion-v1-5" \
    --num_images 54000 \
    --output_dir "./generated_palms" \
    --diverse

echo ""
echo "✅ Complete! You now have 60,000 palm images:"
echo "   - 6,000 original in ./your_6000_palms"
echo "   - 54,000 generated in ./generated_palms"