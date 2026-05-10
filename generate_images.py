"""
Generate images using trained LoRA model
"""

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import argparse
import os
from tqdm import tqdm
import json

class PalmGenerator:
    def __init__(self, base_model, lora_path, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"📦 Loading base model: {base_model}")
        
        # Load pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None
        ).to(device)
        
        # Load LoRA weights
        print(f"🔧 Loading LoRA weights from: {lora_path}")
        self.pipe.load_lora_weights(lora_path)
        
        print("✅ Model loaded successfully!")
    
    def generate(self,
                 prompt="palm",
                 negative_prompt="blurry, deformed, extra fingers, bad anatomy, low quality",
                 num_images=1,
                 num_inference_steps=50,
                 guidance_scale=7.5,
                 seed=None,
                 output_dir="./generated_palms",
                 start_index=0):
        """Generate palm images"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        generated_images = []
        metadata = []
        
        for i in tqdm(range(num_images), desc="Generating images"):
            # Set seed
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed + i)
            else:
                generator = None
            
            # Generate
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                height=512,
                width=512
            ).images[0]
            
            # Save
            filename = f"palm_{start_index + i:06d}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            
            generated_images.append(image)
            
            # Save metadata
            metadata.append({
                "filename": filename,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "seed": seed + i if seed is not None else None,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps
            })
        
        # Save metadata
        with open(os.path.join(output_dir, "generation_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Generated {num_images} images in {output_dir}")
        
        return generated_images
    
    def generate_batch_diverse(self, num_images=1000, output_dir="./generated_palms"):
        """Generate diverse palm images with varied prompts"""
        
        prompts = [
            "palm",
            "detailed palm photograph",
            "human palm, front view",
            "palm of a hand, well lit",
            "clear palm image",
            "hand palm, studio lighting",
            "photograph of palm, high quality"
        ]
        
        images_per_prompt = num_images // len(prompts)
        remainder = num_images % len(prompts)
        offset = 0

        for idx, prompt in enumerate(prompts):
            current_count = images_per_prompt + (1 if idx < remainder else 0)
            self.generate(
                prompt=prompt,
                num_images=current_count,
                output_dir=output_dir,
                seed=1000 + (idx * 100),
                start_index=offset,
            )
            offset += current_count

def main():
    parser = argparse.ArgumentParser(description="Generate palm images using trained LoRA")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to trained LoRA model")
    parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--prompt", type=str, default="palm")
    parser.add_argument("--negative_prompt", type=str, 
                       default="blurry, deformed, extra fingers, bad anatomy, low quality")
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="./generated_palms")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--diverse", action="store_true", help="Generate with diverse prompts")
    
    args = parser.parse_args()
    
    # Create generator
    generator = PalmGenerator(
        base_model=args.base_model,
        lora_path=args.lora_path,
        device=args.device,
    )
    
    # Generate
    if args.diverse:
        generator.generate_batch_diverse(
            num_images=args.num_images,
            output_dir=args.output_dir
        )
    else:
        generator.generate(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_images=args.num_images,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            output_dir=args.output_dir
        )

if __name__ == "__main__":
    main()