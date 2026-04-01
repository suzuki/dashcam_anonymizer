import os
import glob
import json
import cv2
import yaml
import argparse
from ultralytics import YOLO
import shutil
from utils import yolo_to_voc, blur_regions, get_device, get_tracking_config, open_video_writer

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

device = get_device(config["gpu_avail"])
console.print(f"Running on device: {device}", style="bold green")

tracking_config = get_tracking_config(config)

# =========================================================================================
# Search for multiple video file extensions
# =========================================================================================
video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv']
videos = []
for ext in video_extensions:
    videos.extend(glob.glob(os.path.join(config['videos_path'], ext)))
videos = natsorted(videos)

if not(os.path.exists(config["output_folder"])):
    console.print(f"Creating Directory {config['output_folder']} to store the anonymized videos", style="bold green")
    os.mkdir(config["output_folder"])

anonymized_videos_path = config["output_folder"]


def process_video_with_tracking(video_path, output_path, model, config, device,
                                tracker, interpolate_frames):
    """Process a single video using object tracking for temporally consistent detection."""
    video_capture = cv2.VideoCapture(video_path)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_width, frame_height)
    fps = round(video_capture.get(cv2.CAP_PROP_FPS))
    video_capture.release()

    output_video = open_video_writer(output_path, fps, frame_size)

    # Track last known boxes per track ID for interpolation
    last_seen = {}  # track_id -> (frame_num, [x1, y1, x2, y2])
    frame_num = 0

    results = model.track(
        source=video_path,
        stream=True,
        persist=True,
        tracker=tracker,
        conf=config['detection_conf_thresh'],
        device=device,
        verbose=False,
    )

    for result in results:
        frame = result.orig_img
        boxes_to_blur = []

        has_boxes = result.boxes is not None and len(result.boxes) > 0
        has_track_ids = has_boxes and result.boxes.id is not None

        if has_boxes:
            xyxy = result.boxes.xyxy.cpu().tolist()
            track_ids = (
                result.boxes.id.cpu().tolist()
                if has_track_ids
                else [None] * len(xyxy)
            )

            for box, tid in zip(xyxy, track_ids):
                boxes_to_blur.append(box)
                if tid is not None:
                    last_seen[tid] = (frame_num, box)

        # Interpolation: blur regions where tracked objects recently disappeared.
        # Skip interpolation when detections exist but tracker returned no IDs,
        # to avoid blurring stale boxes alongside untracked current detections.
        if interpolate_frames > 0 and not (has_boxes and not has_track_ids):
            active_ids = set()
            if has_track_ids:
                active_ids = set(result.boxes.id.cpu().tolist())

            for tid, (last_frame, last_box) in list(last_seen.items()):
                if tid not in active_ids:
                    gap = frame_num - last_frame
                    if 0 < gap <= interpolate_frames:
                        boxes_to_blur.append(last_box)
                    elif gap > interpolate_frames:
                        del last_seen[tid]

        if boxes_to_blur:
            frame = blur_regions(frame, boxes_to_blur, blur_radius=config["blur_radius"])

        output_video.write(frame)
        frame_num += 1

    output_video.release()
    return frame_num


def process_video_legacy(video_path, output_path, data, config):
    """Process a single video using pre-computed detection JSON (legacy mode)."""
    video_capture = cv2.VideoCapture(video_path)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_width, frame_height)
    fps = round(video_capture.get(cv2.CAP_PROP_FPS))

    output_video = open_video_writer(output_path, fps, frame_size)

    count = 1
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        if str(count) in data:
            frame = blur_regions(frame, data[str(count)], blur_radius=config["blur_radius"])
        output_video.write(frame)
        count += 1

    video_capture.release()
    output_video.release()


if tracking_config["use_tracking"]:
    # =====================================================================
    # Tracking mode: single-pass processing with BoT-SORT/ByteTrack
    # =====================================================================
    console.print(
        f"Processing {len(videos)} videos with tracking "
        f"(tracker={tracking_config['tracker']}, "
        f"interpolate_frames={tracking_config['interpolate_frames']})",
        style="bold green",
    )

    for video in track(videos):
        vid_name, _ = os.path.splitext(os.path.basename(video))
        out_vid_path = osj(anonymized_videos_path, vid_name + '.mp4')

        frame_count = process_video_with_tracking(
            video, out_vid_path, model, config, device,
            tracker=tracking_config["tracker"],
            interpolate_frames=tracking_config["interpolate_frames"],
        )
        console.print(f"Processed {vid_name} ({frame_count} frames with tracking)")

else:
    # =====================================================================
    # Legacy mode: detect → JSON → blur (original 3-stage pipeline)
    # =====================================================================
    if(config["generate_detections"]):
        if os.path.exists("runs"):
            shutil.rmtree("runs")
        console.print("Generating YOLO Detections for the Videos", style="bold green")
        _ = model(source=config['videos_path'],
                save=False,
                save_txt=True,
                conf=config['detection_conf_thresh'],
                device=device,
                project=os.path.join(os.getcwd(), "runs", "detect"),
                name="yolo_videos_pred",
                exist_ok=True)

    if(config["generate_jsons"]):
        print(f"Generating JSONs for {len(videos)} videos")
        for video in track(videos):
            vid_name, _ = os.path.splitext(os.path.basename(video))

            vid = cv2.VideoCapture(video)
            height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
            width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
            vid.release()

            data_dict = {}
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
                                    data_dict[frame_num] = []
                                data_dict[frame_num].append(line)
                if(not os.path.exists("annot_jsons/")):
                    os.mkdir("annot_jsons")
                with open("annot_jsons/"+str(vid_name)+".json", 'w') as f:
                    json.dump(data_dict, f)
            except Exception as e:
                print(f'Could not process annotations for {video}. Error: {e}')

    for video in track(videos):
        vid_name, _ = os.path.splitext(os.path.basename(video))
        out_vid_path = osj(anonymized_videos_path, vid_name + '.mp4')
        json_path = f'annot_jsons/{vid_name}.json'

        if(os.path.exists(json_path)):
            with open(json_path) as F:
                data = json.load(F)
            process_video_legacy(video, out_vid_path, data, config)
            print(f"Processed Video {vid_name}")
        else:
            console.print(f"No objects detected in file {video}, copying file as is.", style="bold orange")
            shutil.copy(video, anonymized_videos_path)
            console.print(f"Copied Video {vid_name}", style="bold green")

    # Clean up temporary files
    if os.path.exists("runs/"):
        console.print(f"Removing Temporary Files...")
        shutil.rmtree("runs/")
    if os.path.exists("annot_jsons/"):
        shutil.rmtree("annot_jsons/")

console.print(f"Blurred Videos are stored in {anonymized_videos_path}", style="bold yellow")
