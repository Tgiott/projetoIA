import os
import pandas as pd
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from tqdm import tqdm
import pybboxes as pbx
import cv2
import matplotlib.pyplot as plt
from random import shuffle
import subprocess

# Ensure the ultralytics package is installed
from ultralytics import YOLO

# Define paths
input_path = 'C:/DEV/projetos/ProjetoIA/kaggle/input/road-sign-detection'
output_path = 'C:/DEV/projetos/ProjetoIA/kaggle/working/yolov5'
annotations_path = os.path.join(input_path, 'annotations')

# Ensure input path exists
if not os.path.exists(input_path):
    raise FileNotFoundError(f"The input path {input_path} does not exist.")

annotations = os.listdir(annotations_path)
if not annotations:
    raise FileNotFoundError(f"No annotations found in {annotations_path}.")

# Initialize lists for dataframe
img_name_list = []
width_list = []
height_list = []
label_list = []
xmin_list = []
ymin_list = []
xmax_list = []
ymax_list = []

# Parse XML annotations
for idx in tqdm(range(len(annotations))):
    tree = ET.parse(os.path.join(annotations_path, annotations[idx]))
    root = tree.getroot()

    img_name = root.find('filename').text
    size = root.find('size')
    width = size.find('width').text
    height = size.find('height').text

    for group in root.findall('object'):
        label = group.find('name').text
        bbox = group.find('bndbox')
        xmin = bbox.find('xmin').text
        ymin = bbox.find('ymin').text
        xmax = bbox.find('xmax').text
        ymax = bbox.find('ymax').text
        
        img_name_list.append(img_name)
        width_list.append(width)
        height_list.append(height)
        xmin_list.append(xmin)
        ymin_list.append(ymin)
        xmax_list.append(xmax)
        ymax_list.append(ymax)
        label_list.append(label)

# Create dataframe
labels_df = pd.DataFrame({
    'img_name': img_name_list,
    'width': width_list,
    'height': height_list,
    'xmin': xmin_list,
    'ymin': ymin_list,
    'xmax': xmax_list,
    'ymax': ymax_list,
    'label': label_list
})

# Map labels to classes
classes = labels_df['label'].unique().tolist()
labels_df['class'] = labels_df['label'].apply(lambda x: classes.index(x))

# Initialize dictionary for YOLO format labels
img_dict = defaultdict(list)

# Convert bounding boxes to YOLO format
for idx in tqdm(range(len(labels_df))):
    sample_label_list = []
    img_name = labels_df.loc[idx, 'img_name']
    xmin = labels_df.loc[idx, 'xmin']
    ymin = labels_df.loc[idx, 'ymin']
    xmax = labels_df.loc[idx, 'xmax']
    ymax = labels_df.loc[idx, 'ymax']
    class_num = labels_df.loc[idx, 'class']
    W, H = int(labels_df.loc[idx, 'width']), int(labels_df.loc[idx, 'height'])

    voc_bbox = (int(xmin), int(ymin), int(xmax), int(ymax))
    x_center, y_center, w, h = pbx.convert_bbox(voc_bbox, from_type="voc", to_type="yolo", image_size=(W, H))

    sample_label_list.append(str(class_num))
    sample_label_list.append(str(x_center))
    sample_label_list.append(str(y_center))
    sample_label_list.append(str(w))
    sample_label_list.append(str(h))
    line = ' '.join(sample_label_list)

    img_dict[img_name].append(line)

# Create labels directory
labels_dir = os.path.join(output_path, 'data', 'labels')
if os.path.exists(labels_dir):
    shutil.rmtree(labels_dir)
os.makedirs(labels_dir, exist_ok=True)

# Generate .txt files for each image
for img_name, lines in img_dict.items():
    img_name = img_name.split('.')[0]
    with open(os.path.join(labels_dir, f'{img_name}.txt'), 'w') as f:
        for line in lines:
            f.write(line + '\n')

# Define directories for train and validation sets
images_path = os.path.join(input_path, 'images')
labels_path = labels_dir
train_dir = os.path.join(output_path, 'data', 'train')
val_dir = os.path.join(output_path, 'data', 'val')

# Create train and validation directories
for dir_path in [train_dir, val_dir]:
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(os.path.join(dir_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dir_path, 'labels'), exist_ok=True)

# Split data into training and validation sets
def split(files, ratio):
    shuffle(files)
    elements = len(files)
    middle = int(elements * ratio)
    return files[:middle], files[middle:]

def copy_files(images_path, labels_path, destination_path, files):
    for file_name in files:
        base_name = file_name.split('.')[0]
        shutil.copy(os.path.join(images_path, f'{base_name}.png'), os.path.join(destination_path, 'images'))
        shutil.copy(os.path.join(labels_path, f'{base_name}.txt'), os.path.join(destination_path, 'labels'))

# Perform the split and copy files
train_ratio = 0.75
files = os.listdir(images_path)
train_files, val_files = split(files, train_ratio)

copy_files(images_path, labels_path, train_dir, train_files)
copy_files(images_path, labels_path, val_dir, val_files)

assert len(os.listdir(os.path.join(train_dir, 'images'))) + len(os.listdir(os.path.join(val_dir, 'images'))) == len(os.listdir(images_path))

# Create the YAML configuration file for YOLOv5
with open(os.path.join(output_path, 'data', 'sign_data.yaml'), 'w') as f:
    f.write('train: ../data/train/images\n')
    f.write('val: ../data/val/images\n')
    f.write('nc: {}\n'.format(len(classes)))
    f.write('names: {}\n'.format(classes))

# Define the training command



