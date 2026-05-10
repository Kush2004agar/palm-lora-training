import csv
import os
import shutil
import argparse

def main():
    parser = argparse.ArgumentParser(description="Filter palmar images from HandInfo.csv")
    parser.add_argument("--base-dir", type=str, required=True, help="Base dataset directory containing HandInfo.csv and Hands/Hands")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for filtered palm dataset")
    args = parser.parse_args()

    base_dir = args.base_dir
    csv_path = os.path.join(base_dir, "HandInfo.csv")
    images_dir = os.path.join(base_dir, "Hands", "Hands")

    output_dir = args.output_dir
    output_csv = os.path.join(output_dir, "PalmInfo.csv")
    output_images_dir = os.path.join(output_dir, "Hands")
    
    os.makedirs(output_images_dir, exist_ok=True)
    
    count_records = 0
    count_images = 0
    
    with open(csv_path, 'r', encoding='utf-8') as f_in, \
         open(output_csv, 'w', encoding='utf-8', newline='') as f_out:
        
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        writer.writeheader()
        
        for row in reader:
            if 'palmar' in row['aspectOfHand'].lower():
                writer.writerow(row)
                count_records += 1
                
                img_name = row['imageName']
                src = os.path.join(images_dir, img_name)
                dst = os.path.join(output_images_dir, img_name)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                    count_images += 1
                    
    print(f"Filtered {count_records} palm records.")
    print(f"Copied {count_images} images to {output_images_dir}.")

if __name__ == '__main__':
    main()
