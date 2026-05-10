"""
Prepare palm dataset for LoRA training
Converts your 6000 images into proper format
"""

import os
import csv
from PIL import Image
from tqdm import tqdm
import argparse
import json

class DatasetPreparer:
    def __init__(self, source_dir, output_dir, resolution=512, caption_style="simple", metadata_csv=None):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.resolution = resolution
        self.caption_style = caption_style
        self.metadata_by_image = self._load_metadata(metadata_csv)

    def _load_metadata(self, metadata_csv):
        if not metadata_csv or not os.path.exists(metadata_csv):
            return {}
        metadata = {}
        with open(metadata_csv, "r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                image_name = row.get("imageName")
                if image_name:
                    metadata[image_name] = row
        return metadata
        
    def prepare(self):
        """Main preparation function"""
        print(f"📂 Preparing dataset from: {self.source_dir}")
        print(f"📁 Output directory: {self.output_dir}")
        
        # Create output directories
        train_dir = os.path.join(self.output_dir, "10_palm")  # 10 = repeats per epoch
        os.makedirs(train_dir, exist_ok=True)
        
        # Get all image files
        image_files = self._get_image_files()
        print(f"📊 Found {len(image_files)} images")
        
        # Process images
        stats = {
            "total": len(image_files),
            "processed": 0,
            "failed": 0,
            "resolution": self.resolution
        }
        
        for idx, img_path in enumerate(tqdm(image_files, desc="Processing images")):
            try:
                self._process_image(img_path, train_dir, idx)
                stats["processed"] += 1
            except Exception as e:
                print(f"\n⚠️ Failed to process {img_path}: {str(e)}")
                stats["failed"] += 1
        
        # Save statistics
        self._save_stats(stats)
        
        print(f"\n✅ Dataset preparation complete!")
        print(f"   Processed: {stats['processed']}")
        print(f"   Failed: {stats['failed']}")
        print(f"   Output: {train_dir}")
        
    def _get_image_files(self):
        """Get all image files from source directory"""
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        image_files = []
        
        for root, dirs, files in os.walk(self.source_dir):
            for file in files:
                if file.lower().endswith(extensions):
                    image_files.append(os.path.join(root, file))
        
        return sorted(image_files)
    
    def _process_image(self, img_path, output_dir, idx):
        """Process single image"""
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Resize
        img = self._resize_image(img)
        
        # Save image
        output_name = f"palm_{idx:06d}.jpg"
        output_path = os.path.join(output_dir, output_name)
        img.save(output_path, quality=95, optimize=True)
        
        # Create caption
        caption = self._generate_caption(img_path, idx)
        caption_path = output_path.replace('.jpg', '.txt')
        with open(caption_path, 'w', encoding='utf-8') as f:
            f.write(caption)
    
    def _resize_image(self, img):
        """Resize image maintaining aspect ratio or center crop"""
        width, height = img.size
        
        # Center crop to square
        if width != height:
            size = min(width, height)
            left = (width - size) // 2
            top = (height - size) // 2
            img = img.crop((left, top, left + size, top + size))
        
        # Resize to target resolution
        if img.size[0] != self.resolution:
            img = img.resize((self.resolution, self.resolution), Image.LANCZOS)
        
        return img
    
    def _generate_caption(self, img_path, idx):
        """Generate caption for image"""
        image_name = os.path.basename(img_path)
        row = self.metadata_by_image.get(image_name)
        if row:
            gender = row.get("gender", "person")
            hand_side = row.get("aspectOfHand", "palmar").split(" ")[0]
            age = row.get("age", "unknown")
            return f"palmar view of a {gender} {hand_side} hand, age {age}, clear palm lines"

        if self.caption_style == "simple":
            return "palm"
        
        elif self.caption_style == "detailed":
            captions = [
                "a detailed photograph of a human palm",
                "palm of a hand, front view",
                "human palm, clear view, well lit",
                "photograph of palm, high quality",
                "detailed palm image, studio lighting"
            ]
            return captions[idx % len(captions)]
        
        elif self.caption_style == "descriptive":
            return "a high-quality photograph of a human palm, front view, well-lit, clear details"
        
        else:
            return "palm"
    
    def _save_stats(self, stats):
        """Save dataset statistics"""
        stats_path = os.path.join(self.output_dir, "dataset_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Prepare palm dataset for LoRA training")
    parser.add_argument("--source", type=str, required=True, help="Source directory with 6000 palm images")
    parser.add_argument("--output", type=str, default="./palm_dataset", help="Output directory")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution (default: 512)")
    parser.add_argument("--caption-style", type=str, default="simple", 
                       choices=["simple", "detailed", "descriptive"],
                       help="Caption generation style")
    parser.add_argument("--metadata-csv", type=str, default=None, help="Optional CSV metadata (PalmInfo.csv)")
    
    args = parser.parse_args()
    
    preparer = DatasetPreparer(
        source_dir=args.source,
        output_dir=args.output,
        resolution=args.resolution,
        caption_style=args.caption_style,
        metadata_csv=args.metadata_csv,
    )
    
    preparer.prepare()

if __name__ == "__main__":
    main()