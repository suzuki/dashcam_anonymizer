import os
import glob
import json
import cv2
import yaml
import argparse
from ultralytics import YOLO
from utils import yolo_to_voc, blur_regions


parser = argparse.ArgumentParser()
parser.add_argument("--config", help = "path of the training configuartion file", required = True)
args = parser.parse_args()

import shutil

if os.path.exists("annot_txt"):
    shutil.rmtree("annot_txt")
if os.path.exists("runs"):
    shutil.rmtree("runs")

#Reading the configuration file
with open(args.config, 'r') as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

model = YOLO(config["model_path"])

device = 'cuda:0' if config["gpu_avail"] else 'cpu'
_ = model(source=config['images_path'],
        save=False,
        save_txt=True,
        conf=config['detection_conf_thresh'],
        device=device,
        name="yolo_images_pred",
        exist_ok=True)


#images = [int(item.split("/")[1].replace(config['img_format'], "")) for item in images]
images = sorted(glob.glob(config['images_path']+"/*"+config["img_format"]))

os.mkdir("annot_txt")

annot_dir = f'runs/detect/yolo_images_pred/labels/'

if os.path.exists(annot_dir):
    for file in os.listdir(annot_dir):
        if file.endswith('.txt'):
            with open(os.path.join(annot_dir, file), 'r') as fin:
                for line in fin.readlines():
                    line = [float(item) for item in line.split()[1:]]
                    line = yolo_to_voc(line, config["img_width"], config["img_height"])
                    data_string = " ".join(str(num) for num in line)
                    with open(f"annot_txt/{file}", "a") as f:
                        f.write(data_string+"\n")
else:
    print("No detections found.")



txt_folder = 'annot_txt/'
image_folder = config['images_path']
output_folder = config['output_folder']

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List all text files in the 'dir' folder
txt_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]

for txt_file in txt_files:
    # Read the text file containing bounding box information
    with open(os.path.join(txt_folder, txt_file), 'r') as f:
        lines = f.readlines()

    # Extract bounding box coordinates from the txt file
    bboxes = []
    for line in lines:
        values = line.strip().split()
        x_min, y_min, x_max, y_max = [int(float(v)) for v in values]
        bboxes.append([x_min, y_min, x_max, y_max])

    # Read the corresponding image
    image_file = txt_file.replace('.txt', config["img_format"])  # Assuming image files have .jpg extension
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)

    # Apply Gaussian blur to each bounding box region
    for bbox in bboxes:
        image = blur_regions(image, bboxes, blur_radius=config['blur_radius'])

    # Save the blurred image to the output folder
    output_file = txt_file.replace('.txt', '_blurred.jpg')
    output_path = os.path.join(output_folder, output_file)
    cv2.imwrite(output_path, image)

print(f"@@ The bluured images are saved in Directory -------> {config['output_folder']}")