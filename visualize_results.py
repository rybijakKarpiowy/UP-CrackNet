import os
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv


images_dir = './inputs_all/test/images/'
true_masks_dir = './inputs_all/test/masks/'
predicted_masks_dir = './outputs_all/'

def visualize_results(image_name):
    print(f'Visualizing results for {image_name}...')
    image_path = os.path.join(images_dir, image_name)
    true_mask_path = os.path.join(true_masks_dir, image_name)
    predicted_mask_path = os.path.join(predicted_masks_dir, image_name).replace('.jpg', '') + '.png'

    image = cv.imread(image_path)
    true_mask = cv.imread(true_mask_path, cv.IMREAD_GRAYSCALE)
    predicted_mask = cv.imread(predicted_mask_path, cv.IMREAD_GRAYSCALE)

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.title('Input Image')
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title('True Mask')
    plt.imshow(true_mask, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title('Predicted Mask')
    plt.imshow(predicted_mask, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
def main():
    image_names = os.listdir(images_dir)
    np.random.seed(12)
    np.random.shuffle(image_names)
    for image_name in image_names:
        visualize_results(image_name)
        
        
if __name__ == '__main__':
    main()