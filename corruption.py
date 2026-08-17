                      
                       
"""
    python add_corruption_to_images.py --input_dir /data/zhangyf/VLN/RxR_we --corruption_type jpeg_compression --corruption_severity 6
"""

import argparse
import os
import glob
import numpy as np
import cv2
from tqdm import tqdm
from make_corruptions.obs_corruption import (
    gaussian_noise, shot_noise, impulse_noise, defocus_blur, motion_blur, 
    zoom_blur, snow, frost, fog, brightness, contrast, elastic_transform, 
    pixelate, jpeg_compression, saturate, occlusion, light_out
)


def corrupt_obs(obs, corruption_type, corruption_severity):
    if corruption_type is None:
        return obs
    if corruption_type == "gaussian_noise":
        obs["rgb"] = gaussian_noise(obs["rgb"], corruption_severity)
    elif corruption_type == "shot_noise":
        obs["rgb"] = shot_noise(obs["rgb"], corruption_severity)
    elif corruption_type == "impulse_noise":
        obs["rgb"] = impulse_noise(obs["rgb"], corruption_severity)
    elif corruption_type == "defocus_blur":
        obs["rgb"] = defocus_blur(obs["rgb"], corruption_severity)
    elif corruption_type == "motion_blur":
        obs["rgb"] = motion_blur(obs["rgb"], corruption_severity)
    elif corruption_type == "zoom_blur":
        obs["rgb"] = zoom_blur(obs["rgb"], corruption_severity)
    elif corruption_type == "snow":
        obs["rgb"] = snow(obs["rgb"], corruption_severity)
    elif corruption_type == "frost":
        obs["rgb"] = frost(obs["rgb"], corruption_severity)
    elif corruption_type == "fog":
        obs["rgb"] = fog(obs["rgb"], corruption_severity)
    elif corruption_type == "brightness":
        obs["rgb"] = brightness(obs["rgb"], corruption_severity)
    elif corruption_type == "contrast":
        obs["rgb"] = contrast(obs["rgb"], corruption_severity)
    elif corruption_type == "elastic_transform":
        obs["rgb"] = elastic_transform(obs["rgb"], corruption_severity)
    elif corruption_type == "pixelate":
        obs["rgb"] = pixelate(obs["rgb"], corruption_severity)
    elif corruption_type == "jpeg_compression":
        obs["rgb"] = jpeg_compression(obs["rgb"], corruption_severity)
    elif corruption_type == "saturate":
        obs["rgb"] = saturate(obs["rgb"], corruption_severity)
    elif corruption_type == "occlusion":
        obs["rgb"] = occlusion(obs["rgb"], corruption_severity)
    elif corruption_type == "light_out":
        obs["rgb"] = light_out(obs["rgb"], corruption_severity)
    obs["rgb"] = obs["rgb"].astype(np.uint8)
    return obs


def process_images(input_dir, output_dir, corruption_type, corruption_severity):
    """
    
    
    Args:
        input_dir: 
        output_dir: 
        corruption_type: 
        corruption_severity: 
    """
            
    os.makedirs(output_dir, exist_ok=True)
    
                    
    episode_folders = []
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path):
            episode_folders.append(item)
    
    episode_folders.sort()
    print(f" {len(episode_folders)} ")
    
                    
    for episode_folder in tqdm(episode_folders, desc=""):
        episode_input_path = os.path.join(input_dir, episode_folder)
        episode_output_path = os.path.join(output_dir, episode_folder)
        
                        
        os.makedirs(episode_output_path, exist_ok=True)
        
                            
        image_files = glob.glob(os.path.join(episode_input_path, "*.jpg"))
        image_files.extend(glob.glob(os.path.join(episode_input_path, "*.png")))
        image_files.sort()
        
                
        for image_file in image_files:
            try:
                      
                image = cv2.imread(image_file)
                if image is None:
                    print(f":  {image_file}")
                    continue
                
                                      
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                      
                obs = {"rgb": image_rgb}
                corrupted_obs = corrupt_obs(obs, corruption_type, corruption_severity)
                corrupted_image = corrupted_obs["rgb"]
                
                              
                corrupted_image_bgr = cv2.cvtColor(corrupted_image, cv2.COLOR_RGB2BGR)
                
                          
                image_filename = os.path.basename(image_file)
                output_image_path = os.path.join(episode_output_path, image_filename)
                cv2.imwrite(output_image_path, corrupted_image_bgr)
                
            except Exception as e:
                print(f":  {image_file} : {str(e)}")
                continue
    
    print(f"\n: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="")
    
    parser.add_argument("--input_dir", type=str, 
                       default="/data/zhangyf/VLN/R2R_we",
                       help="episode")
    parser.add_argument("--corruption_type", type=str, 
                       default="jpeg_compression",
                       choices=["gaussian_noise", "shot_noise", "impulse_noise", 
                               "defocus_blur", "motion_blur", "zoom_blur", 
                               "snow", "frost", "fog", "brightness", "contrast", 
                               "elastic_transform", "pixelate", "jpeg_compression", 
                               "saturate", "occlusion", "light_out"],
                       help="")
    parser.add_argument("--corruption_severity", type=int, 
                       default=6,
                       help=" (1-5)")
    parser.add_argument("--output_dir", type=str, 
                       default=None,
                       help="")
    
    args = parser.parse_args()
    
                     
    if args.output_dir is None:
        args.output_dir = f"{args.input_dir}_{args.corruption_type}"
    
    print(f": {args.input_dir}")
    print(f": {args.output_dir}")
    print(f": {args.corruption_type}")
    print(f": {args.corruption_severity}")
    print("-" * 50)
    
    try:
        process_images(args.input_dir, args.output_dir, args.corruption_type, args.corruption_severity)
    except Exception as e:
        print(f": {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
