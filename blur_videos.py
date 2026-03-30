import os
import glob
import json
import cv2
import yaml
import argparse
from ultralytics import YOLO
import shutil
from utils import yolo_to_voc, blur_regions

from rich.console import Console
from rich.progress import track
from natsort import natsorted
from os.path import join as osj

parser = argparse.ArgumentParser()
parser.add_argument("--config", help = "path of the training configuartion file", required = True)
args = parser.parse_args()
console = Console()


console.print(f"Reading the Configuration file from {args.config}", style="bold green")
with open(args.config, 'r') as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)


console.print("Loading YOLO Model...", style="bold green")
model = YOLO(config["model_path"])

if(config["generate_detections"]):
    if os.path.exists("runs"):
        shutil.rmtree("runs")
    console.print("Generating YOLO Detections for the Videos", style="bold green")
    device = 'cuda:0' if config["gpu_avail"] else 'cpu'
    console.print(f"Running on {'GPU' if config['gpu_avail'] else 'CPU'}", style="bold green")
    _ = model(source=config['videos_path'],
            save=False,
            save_txt=True,
            conf=config['detection_conf_thresh'],
            device=device,
            name="yolo_videos_pred",
            exist_ok=True)
    
# =========================================================================================
# CHANGE 1: Search for multiple video file extensions, not just .mp4
# =========================================================================================
video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv'] # Add any other video formats you use
videos = []
for ext in video_extensions:
    videos.extend(glob.glob(os.path.join(config['videos_path'], ext)))

videos = natsorted(videos)
# =========================================================================================

if(config["generate_jsons"]):
    print(f"Generating JSONs for {len(videos)} videos")
    for video in track(videos):
        # =========================================================================================
        # CHANGE 2: Use os.path.splitext to robustly get the video name without its extension
        # =========================================================================================
        vid_name, _ = os.path.splitext(os.path.basename(video))
        # =========================================================================================

        vid = cv2.VideoCapture(video)
        height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        vid.release() # Release the video capture object after getting dimensions
        
        data_dict = {}
        # The yolo prediction folder name matches the video name without extension
        annot_dir = natsorted(glob.glob(f'runs/detect/yolo_videos_pred/labels/{vid_name}_*.txt'))
        
        try:
            for file in annot_dir:
                if (os.path.basename(file).endswith('.txt')):
                    frame_num = int(os.path.basename(file).replace(".txt","").split("_")[1])
                    with open(file, 'r') as fin:
                        for line in fin.readlines():
                            line = [float(item) for item in line.split()[1:]]
                            line = yolo_to_voc(line, width, height)
                            if(frame_num not in data_dict.keys()):
                                data_dict[frame_num] = [] # Initialize as empty list
                            data_dict[frame_num].append(line)
            if(not os.path.exists("annot_jsons/")):
                os.mkdir("annot_jsons")
            with open("annot_jsons/"+str(vid_name)+".json", 'w') as f:
                json.dump(data_dict, f)
        except Exception as e:
            print(f'Could not process annotations for {video}. Error: {e}')




if not(os.path.exists(config["output_folder"])):
    console.print(f"Creating Directory {config['output_folder']} to store the anonymized videos", style="bold green")
    os.mkdir(config["output_folder"])

anonymized_videos_path = config["output_folder"]

for video in track(videos):
    # =========================================================================================
    # CHANGE 3: Use os.path.splitext again for consistency
    # =========================================================================================
    vid_name, _ = os.path.splitext(os.path.basename(video))
    # =========================================================================================
    
    json_path = f'annot_jsons/{vid_name}.json'
    if(os.path.exists(json_path)):
        with open(json_path) as F:
            data = json.load(F)

            video_capture = cv2.VideoCapture(video)
            
            # =========================================================================================
            # CHANGE 4: The output video will always be an MP4 for compatibility with the 'avc1' codec.
            # =========================================================================================
            out_vid_path = osj(anonymized_videos_path, vid_name + '.mp4')
            # =========================================================================================
            
            frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_size = (frame_width, frame_height)
            
            fps = round(video_capture.get(cv2.CAP_PROP_FPS))
            # Try H.264 first, fall back to MPEG-4 if unavailable
            for codec in ['avc1', 'mp4v']:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                output_video = cv2.VideoWriter(out_vid_path, fourcc, fps, frame_size)
                if output_video.isOpened():
                    break
                output_video.release()
            count = 1
            while True:
                ret, frame = video_capture.read()
                
                if not ret:
                    break
                
                if str(count) in data:
                    frame = blur_regions(frame, data[str(count)], blur_radius=config["blur_radius"])

                output_video.write(frame)
                count+=1
            video_capture.release()
            output_video.release()
        print(f"Processed Video {vid_name}")
    else:
        console.print(f"No objects detected in file {video}, copying file as is.", style="bold orange")
        shutil.copy(video, anonymized_videos_path)
        console.print(f"Copied Video {vid_name}", style="bold green")
        
#remove runs folder
if os.path.exists("runs/"):
    console.print(f"Removing Temporary Files...")
    shutil.rmtree("runs/")
if os.path.exists("annot_jsons/"):
    shutil.rmtree("annot_jsons/")

console.print(f"Blurred Videos are stored in {anonymized_videos_path}", style="bold yellow")
