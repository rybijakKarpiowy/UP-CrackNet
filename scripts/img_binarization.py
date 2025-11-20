# Add src to path
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multiprocessing import Pool
from classifier import CrackClassifier
import cv2 as cv
import time
import numpy as np

root_dir = './model_crack500_results/best/'
out_dir = './outputs_all/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)


def bilater_Otsu(img_path):
    img = cv.imread(os.path.join(root_dir, img_path))
    img_bilater = cv.bilateralFilter(img, 25, 450, 15)
    img_bilater = cv.cvtColor(img_bilater, cv.COLOR_BGR2GRAY)
    _, Otsu_map = cv.threshold(img_bilater, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    new_path = os.path.join(out_dir, img_path)
    cv.imwrite(new_path, Otsu_map)
    
def bilater_adaptive_thresh(img_path):
    img = cv.imread(os.path.join(root_dir, img_path))
    img_bilater = cv.bilateralFilter(img, 25, 450, 15)
    img_bilater = cv.cvtColor(img_bilater, cv.COLOR_BGR2GRAY)
    adaptive_map = cv.adaptiveThreshold(img_bilater, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv.THRESH_BINARY, 11, 2)
    new_path = os.path.join(out_dir, img_path)
    cv.imwrite(new_path, adaptive_map)
    
def bilater_otsu_w_classifier(img_path: str):
    classifier_path = './saved-model/crack_classifier_resnet18.pth'
    model = CrackClassifier(model_path=classifier_path)
    
    # Use classifier to decide whether to apply Otsu
    test_dir = './crack_segmentation_dataset/test/images/'
    original_file = ""
    for filename in os.listdir(test_dir):
        if filename.replace('.jpg', '') == img_path.replace('.png', ''):
            original_file = filename
            break
    if original_file == "":
        print(f"Original file for {img_path} not found.")
        return
    label = model.predict(os.path.join(test_dir, original_file))
    
    if label == 0:  # non-crack
        # return empty mask
        Otsu_map = np.zeros((448, 448), dtype=np.uint8)
        
    else:  # crack
        img = cv.imread(os.path.join(root_dir, img_path))
        img_bilater = cv.bilateralFilter(img, 25, 450, 15)
        img_bilater_gray = cv.cvtColor(img_bilater, cv.COLOR_BGR2GRAY)
        _, Otsu_map = cv.threshold(img_bilater_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        
    new_path = os.path.join(out_dir, img_path)
    cv.imwrite(new_path, Otsu_map)
    
def bilater_otsu_w_edges(img_path, distance_threshold=0.06):
    img = cv.imread(os.path.join(root_dir, img_path))
    img_bilater = cv.bilateralFilter(img, 25, 450, 15)
    img_bilater_gray = cv.cvtColor(img_bilater, cv.COLOR_BGR2GRAY)
    _, Otsu_map = cv.threshold(img_bilater_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    # Otsu map generates large flat areas that are far away from edges
    # We can use edges to refine the Otsu map by calculating distance transform
    edges = cv.Canny(img_bilater_gray, 100, 200)
    distance = cv.distanceTransform(255 - edges, cv.DIST_L2, 5)
    distance = cv.normalize(distance, None, 0, 1.0, cv.NORM_MINMAX)
    refined_map = np.where(distance < distance_threshold, 0, Otsu_map)  # threshold distance to refine
    
    new_path = os.path.join(out_dir, img_path)
    cv.imwrite(new_path, refined_map)
    
def bilater_otsu_w_edges_and_classifier(img_path, distance_threshold=0.05):
    classifier_path = './saved-model/crack_classifier_resnet18.pth'
    model = CrackClassifier(model_path=classifier_path)
    
    # Use classifier to decide whether to apply Otsu
    test_dir = './crack_segmentation_dataset/test/images/'
    original_file = ""
    for filename in os.listdir(test_dir):
        if filename.replace('.jpg', '') == img_path.replace('.png', ''):
            original_file = filename
            break
    if original_file == "":
        print(f"Original file for {img_path} not found.")
        return
    label = model.predict(os.path.join(test_dir, original_file))
    
    if label == 0:  # non-crack
        # return empty mask
        mask = np.zeros((448, 448), dtype=np.uint8)
    else:  # crack
        img = cv.imread(os.path.join(root_dir, img_path))
        img_bilater = cv.bilateralFilter(img, 25, 450, 15)
        img_bilater_gray = cv.cvtColor(img_bilater, cv.COLOR_BGR2GRAY)
        _, Otsu_map = cv.threshold(img_bilater_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        
        # Otsu map generates large flat areas that are far away from edges
        # We can use edges to refine the Otsu map by calculating distance transform
        edges = cv.Canny(img_bilater_gray, 100, 200)
        distance = cv.distanceTransform(255 - edges, cv.DIST_L2, 5)
        distance = cv.normalize(distance, None, 0, 1.0, cv.NORM_MINMAX)
        mask = np.where(distance < distance_threshold, 0, Otsu_map)  # threshold distance to refine
        
    new_path = os.path.join(out_dir, img_path)
    cv.imwrite(new_path, mask)
    
def bilater_otsu_after_combining_w_edges(img_path, power=0.8):
    img = cv.imread(os.path.join(root_dir, img_path))
    img_bilater = cv.bilateralFilter(img, 25, 450, 15)
    img_bilater_gray = cv.cvtColor(img_bilater, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(img_bilater_gray, 100, 200)
    distance = cv.distanceTransform(255 - edges, cv.DIST_L2, 5)
    distance = cv.normalize(distance, None, 0, 1.0, cv.NORM_MINMAX)
    # Power transform to enhance distance effect
    # PT
    distance = np.power(distance, power)
    distance = cv.normalize(distance, None, 0, 1.0, cv.NORM_MINMAX)
    
    # Modulate bilateral image with distance to edges
    combined = img_bilater_gray.astype(np.float32) * (1 - distance)
    combined = np.clip(combined, 0, 255).astype(np.uint8)
    
    _, Otsu_map = cv.threshold(combined, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    new_path = os.path.join(out_dir, img_path)
    cv.imwrite(new_path, Otsu_map)
    
    
def bilater_otsu_after_combining_w_edges_and_classifier(img_path, power=0.8):
    classifier_path = './saved-model/crack_classifier_resnet18.pth'
    model = CrackClassifier(model_path=classifier_path)
    
    # Use classifier to decide whether to apply Otsu
    test_dir = './crack_segmentation_dataset/test/images/'
    original_file = ""
    for filename in os.listdir(test_dir):
        if filename.replace('.jpg', '') == img_path.replace('.png', ''):
            original_file = filename
            break
    if original_file == "":
        print(f"Original file for {img_path} not found.")
        return
    label = model.predict(os.path.join(test_dir, original_file))
    
    if label == 0:  # non-crack
        # return empty mask
        mask = np.zeros((448, 448), dtype=np.uint8)
    else:  # crack
        img = cv.imread(os.path.join(root_dir, img_path))
        img_bilater = cv.bilateralFilter(img, 25, 450, 15)
        img_bilater_gray = cv.cvtColor(img_bilater, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(img_bilater_gray, 100, 200)
        distance = cv.distanceTransform(255 - edges, cv.DIST_L2, 5)
        distance = cv.normalize(distance, None, 0, 1.0, cv.NORM_MINMAX)
        # Power transform to enhance distance effect
        # PT
        distance = np.power(distance, power)
        distance = cv.normalize(distance, None, 0, 1.0, cv.NORM_MINMAX)
        
        # Modulate bilateral image with distance to edges
        combined = img_bilater_gray.astype(np.float32) * (1 - distance)
        combined = np.clip(combined, 0, 255).astype(np.uint8)
        
        _, mask = cv.threshold(combined, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        
    new_path = os.path.join(out_dir, img_path)
    cv.imwrite(new_path, mask)

    
def main():
    list_image = os.listdir(root_dir)
    print(list_image)
    workers = os.cpu_count()
    # number of processors used will be equal to workers
    with Pool(workers) as p:
        # p.map(bilater_img, list_image)
        p.map(bilater_otsu_after_combining_w_edges_and_classifier, list_image)

if __name__ == '__main__':
    time1 = time.time()
    main()
    time2 = time.time()
    print("time_cost is {}".format(time2 - time1))
