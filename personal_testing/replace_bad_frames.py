import os
import cv2
import numpy as np
from glob import glob
from pathlib import Path
import argparse

def replace_initial_frames(episode_num, n_frames):
    # Format episode number with leading zeros
    episode_str = f"{episode_num:06d}"
    
    # Define the camera views
    views = ['claw', 'front', 'phone']
    
    for view in views:
        # Construct the path pattern
        path_pattern = f"data/cmcgartoll/banana-grama-bot/videos/observation.images.{view}_episode_{episode_str}/frame_*.png"
        image_files = sorted(glob(path_pattern))
        
        if not image_files:
            print(f"No images found for {view} view in episode {episode_str}")
            continue
            
        if len(image_files) <= n_frames:
            print(f"Warning: {view} view has fewer than {n_frames + 1} frames")
            continue
        
        # Read the n+1th frame that will be used as replacement
        replacement_frame = cv2.imread(image_files[n_frames])
        
        # Replace first n frames with copy of n+1th frame
        for i in range(n_frames):
            cv2.imwrite(image_files[i], replacement_frame)
            
        print(f"Processed {view} view: replaced first {n_frames} frames")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Replace initial frames in episode image sequences')
    parser.add_argument('episode_num', type=int, help='Episode number to process')
    parser.add_argument('n_frames', type=int, help='Number of frames to replace')
    
    args = parser.parse_args()
    replace_initial_frames(args.episode_num, args.n_frames)