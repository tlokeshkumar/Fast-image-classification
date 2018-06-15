import numpy as np 
import shutil
from glob import glob 
import argparse
import os
import random
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help = "Enter the dataset path")
parser.add_argument("--val_split",type = float, help = "Enter the val split (fraction of dataset that goes to validation")
args = parser.parse_args()
'''
IF YOU WANT TO RUN THIS PROGRAM AS IT IS WITHOUT CHANGING ANYTHING, THEN THIS SHOULD BE THE DIRECTORY
STRUCTURE

train_val_split.py
dataset_folder  \
                    class_1_folder  \
                                        *.jpg
                    class_2_folder  \
                                        *.jpg
                    .
                    .
                    .

Sample execution
python train_val_split.py --dataset dataset_folder --val_split 0.2
'''
train_folders = glob(args.dataset + "/*")
class_names = [os.path.split(x)[1] for x in train_folders]
train_dataset = glob(args.dataset + "/**/*")
TRAIN_DIR = "train_dir"
VAL_DIR = "val_dir"
def create_class_folders(dir_, class_names):
    for class_name in class_names:
        if not os.path.exists(dir_ +"/"+class_name):
            os.mkdir(dir_ +"/"+class_name)

if not os.path.exists(TRAIN_DIR):
    os.mkdir(TRAIN_DIR)
if not os.path.exists(VAL_DIR):
    os.mkdir(VAL_DIR)

create_class_folders(TRAIN_DIR, class_names)
create_class_folders(VAL_DIR, class_names)

random.shuffle(train_dataset)
shuffled_dataset = train_dataset
length_dataset = len(train_dataset)
no_val_examples = int(args.val_split*length_dataset)
train_examples = shuffled_dataset[:-no_val_examples]
val_examples = shuffled_dataset[-no_val_examples:]

for train in tqdm(train_examples):
    # Please change this to your custom use please think !!!!
    class_rep = os.path.split(os.path.split(train)[0])[1]
    image_name = os.path.split(train)[1]
    dest = TRAIN_DIR+"/"+class_rep+"/"+image_name
    shutil.move(train, dest)

for val in tqdm(val_examples):
    # Please change this to your custom use please think!!
    class_rep = os.path.split(os.path.split(val)[0])[1]
    image_name = os.path.split(val)[1]
    dest = VAL_DIR+"/"+class_rep+"/"+image_name
    shutil.move(val, dest)

os.system('rm -r '+args.dataset)