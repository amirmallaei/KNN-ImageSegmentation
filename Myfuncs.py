import numpy as np

def my_mean(mask, img):
    mask_size = mask.shape;
    count = 0; jam = 0;
    for i in range(0,mask_size[0]):
        for j in range(0,mask_size[1]):
            if mask[i,j] > 0:
                count += 1
                jam += img[i,j]
    return jam/count
