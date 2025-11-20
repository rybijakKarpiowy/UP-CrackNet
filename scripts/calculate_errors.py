import os
import numpy as np
from sklearn.metrics import jaccard_score, f1_score, accuracy_score
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='Calculate segmentation errors')
parser.add_argument('approach_name', type=str, help='Name of the segmentation approach')
args = parser.parse_args()

# F1 IoU Pr Re Acc
def calculate_errors(pred_dir, gt_dir):
    pred_files = sorted(os.listdir(pred_dir))
    gt_files = sorted(os.listdir(gt_dir))

    total_f1 = 0.0
    total_iou = 0.0
    total_pr = 0.0
    total_re = 0.0
    total_acc = 0.0
    num_images = len(pred_files)

    for pred_file, gt_file in zip(pred_files, gt_files):
        pred_path = os.path.join(pred_dir, pred_file)
        gt_path = os.path.join(gt_dir, gt_file)

        pred_img = np.array(Image.open(pred_path).convert('L')).flatten() // 255
        gt_img = np.array(Image.open(gt_path).convert('L')).flatten() // 255
        
        if np.sum(gt_img) == 0 and np.sum(pred_img) == 0:
            total_f1 += 1.0
            total_iou += 1.0
        else:
            total_f1 += f1_score(gt_img, pred_img)
            total_iou += jaccard_score(gt_img, pred_img)
            
        total_acc += accuracy_score(gt_img, pred_img)

        tp = np.sum((pred_img == 1) & (gt_img == 1))
        fp = np.sum((pred_img == 1) & (gt_img == 0))
        fn = np.sum((pred_img == 0) & (gt_img == 1))

        pr = tp / (tp + fp) if fp > 0 else 1.0
        re = tp / (tp + fn) if fn > 0 else 1.0

        total_pr += pr
        total_re += re

    avg_f1 = total_f1 / num_images
    avg_iou = total_iou / num_images
    avg_pr = total_pr / num_images
    avg_re = total_re / num_images
    avg_acc = total_acc / num_images

    return avg_f1, avg_iou, avg_pr, avg_re, avg_acc

if __name__ == '__main__':
    approach_name = args.approach_name
    pred_directory = './outputs_all/'
    gt_directory = './crack_segmentation_dataset/test/masks/'

    f1, iou, pr, re, acc = calculate_errors(pred_directory, gt_directory)
    
    results_path = f'./results.csv'
    with open(results_path, 'a') as f:
        f.write(f'{approach_name},{f1:.4f},{iou:.4f},{pr:.4f},{re:.4f},{acc:.4f}\n')