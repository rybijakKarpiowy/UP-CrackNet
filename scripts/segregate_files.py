# move 20% of train data to val folder
import os
import shutil
import random
import numpy as np

src_dir = './crack_segmentation_dataset/train/images'
train_dir = './crack_segmentation_dataset/noncrack/train/images'
val_dir = './crack_segmentation_dataset/noncrack/validation/images'

noncrack_filenames = [x for x in os.listdir(src_dir) if x.startswith('noncrack')]
print(f"Total noncrack files: {len(noncrack_filenames)}")

np.random.seed(42)
noncrack_filenames = np.array(noncrack_filenames)
np.random.shuffle(noncrack_filenames)

train_filenames = noncrack_filenames[int(0.2 * len(noncrack_filenames)):]
val_filenames = noncrack_filenames[:int(0.2 * len(noncrack_filenames))]
print(f"Train noncrack files: {len(train_filenames)}")
print(f"Val noncrack files: {len(val_filenames)}")

for filename in train_filenames:
    src_path = os.path.join(src_dir, filename)
    dst_path = os.path.join(train_dir, filename)
    # Copy file from train to src_dir
    shutil.copy(src_path, dst_path)
    
for filename in val_filenames:
    src_path = os.path.join(src_dir, filename)
    dst_path = os.path.join(val_dir, filename)
    # Copy file from src_dir to val_dir
    shutil.copy(src_path, dst_path)