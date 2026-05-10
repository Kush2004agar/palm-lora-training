$ErrorActionPreference = "Stop"

Write-Host "Starting Palm LoRA Training Pipeline"
Write-Host "========================================"

# Set encoding to avoid emoji errors
$env:PYTHONIOENCODING="utf-8"

# Step 1: Prepare dataset
Write-Host "`nStep 1: Preparing dataset..."
python prepare_dataset.py --source "./Hands" --output "./palm_dataset" --resolution 512 --caption-style simple
if ($LASTEXITCODE -ne 0) { throw "Dataset preparation failed." }

# Step 2: Train LoRA
Write-Host "`nStep 2: Training LoRA..."
python train_lora.py --data_dir "./palm_dataset/10_palm" --output_dir "./output" --pretrained_model "runwayml/stable-diffusion-v1-5" --resolution 512 --batch_size 2 --num_epochs 25 --learning_rate 1e-4 --lora_rank 32 --lora_alpha 32 --mixed_precision fp16 --enable_latent_cache
if ($LASTEXITCODE -ne 0) { throw "LoRA training failed." }

# Step 3: Generate images
Write-Host "`nStep 3: Generating 54,000 new palm images..."
python generate_images.py --lora_path "./output/checkpoint-final" --base_model "runwayml/stable-diffusion-v1-5" --num_images 54000 --output_dir "./generated_palms" --diverse
if ($LASTEXITCODE -ne 0) { throw "Image generation failed." }

Write-Host "`nComplete! You now have 60,000 palm images."
