
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3


# In[ ]:


import cv2
from readImages import *
import math
import numpy as np 
from glob import glob
import os


# In[ ]:


IN_FILE_DIR = '../faces/'
OUT_FILE_DIR = '../data/'


# In[ ]:


def rotatedRectWithMaxArea(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    degree), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """
    if w <= 0 or h <= 0:
        return 0,0
    # convert to radian
    angle = angle / 180.0 * math.pi

    width_is_longer = w >= h
    side_long, side_short = (w,h) if width_is_longer else (h,w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5*side_short
        wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a*cos_a - sin_a*sin_a
        wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

    return wr,hr


# In[ ]:


def rotate(img, minAngle = 1, maxAngle = 50, toshow=False):
    angle = np.random.uniform(minAngle, maxAngle) # Unit: degree
    height, width = img.shape[:2]
    maxHeight, maxWidth = rotatedRectWithMaxArea(width, height, angle)
    
    # Rotate Image
    center = ( height // 2, width // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (width, height))
    
    h_min = center[0] - int(maxHeight * 0.5)
    h_max = center[0] + int(maxHeight * 0.5)
    w_min = center[1] - int(maxWidth * 0.5)
    w_max = center[1] + int(maxWidth * 0.5)
    
    rotated = rotated[h_min : h_max + 1, w_min : w_max + 1]
    rotated = cv2.resize(rotated, img.shape[:2], interpolation=cv2.INTER_CUBIC)
    
    if toshow:
        cv2.imshow('rotated {}'.format(angle), rotated)
        
    return rotated


# In[ ]:


def flip(img, toshow=False):
    horizontal_img = img.copy()
    vertical_img = img.copy()
    both_img = img.copy()
    
    horizontal_img = cv2.flip(img, 0)
    vertical_img = cv2.flip(img, 1)
    both_img = cv2.flip(img, -1)
    
    if toshow:
        cv2.imshow("horizontal", horizontal_img)
        cv2.imshow("vertical", vertical_img)
        cv2.imshow("both flip", both_img)
        
    return (horizontal_img, vertical_img, both_img)


# In[ ]:


def augmentation(img):
    return [img, rotate(img)] + list(flip(img))


# In[ ]:


if __name__ == '__main__':
    files = glob(os.path.join(IN_FILE_DIR, '*.jpg'))
    print('  * Starting Data Augmentation ...')
    for file in files:
        img_origin = cv2.imread(file)
        imgs = augmentation(img_origin)
        
        for i, img in enumerate(imgs):
            cv2.imwrite(os.path.join(OUT_FILE_DIR, file + '_' + str(i)), img)
        
    print('  * DONE!!')

