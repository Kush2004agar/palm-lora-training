# Architecture Upgrade Roadmap

This project currently fine-tunes Stable Diffusion v1.5 with LoRA on palm images.

## Current Adaptation Scope

- LoRA targets attention projections plus feed-forward projections:
  - `to_k`, `to_q`, `to_v`, `to_out.0`
  - `ff.net.0.proj`, `ff.net.2`

## Planned Upgrade Paths

### 1. SDXL LoRA (Recommended first)

- Base model: `stabilityai/stable-diffusion-xl-base-1.0`
- Resolution: 768 or 1024
- Keep LoRA-only finetuning for lower VRAM

### 2. StyleGAN2-ADA (Alternative if strict GAN workflow is required)

- Best for small datasets with adaptive augmentation
- Requires dedicated training/inference pipeline separate from Diffusers

### 3. Full Diffusion Finetune (High compute)

- DreamBooth/Full UNet finetuning with prior preservation
- Higher fidelity but significantly higher compute and storage costs

## Quality Validation Gates

- FID trend improves over checkpoints
- No memorization spikes in nearest-neighbor checks
- Stable CLIP-score for prompt alignment
