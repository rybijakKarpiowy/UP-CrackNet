import cv2
import numpy as np
import random
import torch

def random_checkboard_mask_new(img, ratio_n=None):

    if ratio_n == None:
        random_value = torch.rand(1)

    if random_value < 1/8:
        mask = np.load("./ck_mask/ck_0.npy")

    elif random_value < 2/8:
        mask = np.load("./ck_mask/ck_1.npy")
        
    elif random_value < 3/8:
        mask = np.load("./ck_mask/ck_2.npy")
        
    elif random_value < 4/8:
        mask = np.load("./ck_mask/ck_3.npy")
        
    elif random_value < 5/8:
        mask = np.load("./ck_mask/ck_4.npy")
        
    elif random_value < 6/8:
        mask = np.load("./ck_mask/ck_5.npy")
        
    elif random_value < 7/8:
        mask = np.load("./ck_mask/ck_6.npy")
        
    else:
        mask = np.load("./ck_mask/ck_7.npy")
        
    return mask

if __name__ == '__main__':
    img_path = "/media/nachuan/TOSHIBA_nachuan/Conditional GAN/0915/crack500_test_ori/a/20160222_114759_1281_721.png"

    img = cv2.imread(img_path)
    print(img.shape)

   


