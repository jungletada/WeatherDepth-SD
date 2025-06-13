# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse
import numpy as np
import PIL.Image as pil
from tqdm import tqdm
from utils import readlines
from kitti_utils import generate_depth_map


def export_gt_depths_kitti():
    parser = argparse.ArgumentParser(description='export_gt_depth')
    parser.add_argument('--data_path',
                        type=str,
                        default='data/kitti/rgb',
                        help='path to the root of the KITTI data')
    parser.add_argument('--split',
                        type=str,
                        help='which split to export gt from',
                        default="eigen_zhou",
                        choices=["eigen_zhou", "eigen_raw", "eigen_full", "eigen_benchmark"])
    
    opt = parser.parse_args()
    split_folder = os.path.join(os.path.dirname(__file__), "splits", opt.split)
    
    split_types = ['test', 'val']
    
    def generate_npy_label(lines):
        gt_depths = {}
        
        for idx, line in tqdm(enumerate(lines), total=len(lines), desc="Generating depth maps"):
            folder, frame_id, _ = line.split()
            frame_id = int(frame_id)
            if opt.split != "eigen_benchmark":
                calib_dir = os.path.join(opt.data_path, folder.split("/")[0])
                velo_filename = os.path.join(opt.data_path, folder.split("/")[1],
                                            "velodyne_points/data", "{:010d}.bin".format(frame_id))
                gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True)
                
            else:
                gt_depth_path = os.path.join(opt.data_path, folder, "proj_depth",
                                            "groundtruth", "image_02", "{:010d}.png".format(frame_id))
                gt_depth = np.array(pil.open(gt_depth_path)).astype(np.float32) / 256
            # gt_depths.append(gt_depth.astype(np.float32))
            gt_depths[str(idx)] = gt_depth.astype(np.float32)
            # print(idx)
        output_path = os.path.join(split_folder, f"{split_type}_gt_depths.npz")
        print("Saving to {}".format(opt.split))
        # Convert list to numpy array with consistent shape
        # gt_depths = np.stack(gt_depths)
        # Save dictionary with individual arrays
        np.savez_compressed(output_path, **gt_depths)
    
    print("Exporting ground truth depths for {}".format(opt.split))
    
    for split_type in split_types:
        print(f"Processing for {split_type}_files.txt")
        lines = readlines(os.path.join(split_folder, f"{split_type}_files.txt"))
        generate_npy_label(lines)
    

if __name__ == "__main__":
    export_gt_depths_kitti()