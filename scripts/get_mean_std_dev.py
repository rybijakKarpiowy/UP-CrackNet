# Calculate mean and std dev of a dataset
import os
import cv2
import numpy as np

def calculate_mean_std_dev(image_dir):
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
    mean = np.zeros(3)
    std_dev = np.zeros(3)
    num_images = len(image_files)

    for i, img_file in enumerate(image_files):
        print(f'Processing image {i+1}/{num_images}', end='\r')
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path) / 255.0  # Normalize to [0, 1]
        mean += np.mean(img, axis=(0, 1))
        std_dev += np.std(img, axis=(0, 1))

    mean /= num_images
    std_dev /= num_images

    return mean.tolist(), std_dev.tolist()

if __name__ == '__main__':
    image_directory = './crack_segmentation_dataset/train/images/'
    mean, std_dev = calculate_mean_std_dev(image_directory)
    print(f'Mean: {mean}')
    print(f'Standard Deviation: {std_dev}')