import argparse
import json
import os
import numpy as np
from typing import Type

from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import face_detector
from face_detector import VideoDataset
from face_detector import VideoFaceDetector
from utils import get_video_paths, get_method
import argparse


def process_videos(videos, detector_cls: Type[VideoFaceDetector], selected_dataset, opt):
    detector = face_detector.__dict__[detector_cls](device="cuda:0")

    dataset = VideoDataset(videos)
    
    loader = DataLoader(dataset, shuffle=False, num_workers=opt.processes, batch_size=1, collate_fn=lambda x: x)
    missed_videos = []
    for item in tqdm(loader): 
        result = {}
        video, indices, frames = item[0]
        if selected_dataset == 1:
            method = get_method(video, opt.data_path)
            out_dir = os.path.join(opt.data_path, "boxes", method)
        else:
            out_dir = os.path.join(opt.data_path, "boxes")

        id = os.path.splitext(os.path.basename(video))[0]

        if os.path.exists(out_dir) and "{}.json".format(id) in os.listdir(out_dir):
            continue
        batches = [frames[i:i + detector._batch_size] for i in range(0, len(frames), detector._batch_size)]
      
        for j, frames in enumerate(batches):
            result.update({int(j * detector._batch_size) + i : b for i, b in zip(indices, detector._detect_faces(frames))})
        
       
        os.makedirs(out_dir, exist_ok=True)
        print(len(result))
        if len(result) > 0:
            with open(os.path.join(out_dir, "{}.json".format(id)), "w") as f:
                json.dump(result, f)
        else:
            missed_videos.append(id)

    if len(missed_videos) > 0:
        print("The detector did not find faces inside the following videos:")
        print(id)
        print("We suggest to re-run the code decreasing the detector threshold.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="DFDC", type=str,
                        help='Dataset (DFDC / FACEFORENSICS)')
    parser.add_argument('--data_path', default='', type=str,
                        help='Videos directory')
    parser.add_argument("--detector-type", help="Type of the detector", default="FacenetDetector",
                        choices=["FacenetDetector"])
    parser.add_argument("--processes", help="Number of processes", default=1)
    opt = parser.parse_args()
    print(opt)


    if opt.dataset.upper() == "DFDC":
        dataset = 0
    else:
        dataset = 1

    videos_paths = []
    if dataset == 1:
        videos_paths = get_video_paths(opt.data_path, dataset)
    else:
        os.makedirs(os.path.join(opt.data_path, "boxes"), exist_ok=True)
        already_extracted = os.listdir(os.path.join(opt.data_path, "boxes"))
        for folder in os.listdir(opt.data_path):
            if "boxes" not in folder and "zip" not in folder:
                if os.path.isdir(os.path.join(opt.data_path, folder)): # For training and test set
                    for video_name in os.listdir(os.path.join(opt.data_path, folder)):
                        if video_name.split(".")[0] + ".json" in already_extracted:
                            continue
                        videos_paths.append(os.path.join(opt.data_path, folder, video_name))
                else: # For validation set
                    videos_paths.append(os.path.join(opt.data_path, folder))

    process_videos(videos_paths, opt.detector_type, dataset, opt)


if __name__ == "__main__":
    main()
