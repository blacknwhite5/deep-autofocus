"""
@author H. Kim, M. Oh
@company EgoVid Inc
"""
import numpy as np
import math
import cv2

def _mean_intensity(img):
    return np.sum(img, dtype=np.int32) / img.size

def _center_crop(img, crop_size):
    h, w = img.shape
    cntr_h, cntr_w = h//2, w//2
    crop_h, crop_w = crop_size[0]//2, crop_size[1]//2
    img = img[cntr_h-crop_h:cntr_h+crop_h, cntr_w-crop_w:cntr_w+crop_w]
    return img
    
def var_abs(img_name, crop_size=None):
    img = cv2.imread(img_name,0)
    if crop_size is not None:
        img = _center_crop(img, crop_size)
    mean = _mean_intensity(img)
    var_intensity = np.abs(img - mean)
    var_intensity = int(np.sum(var_intensity) / img.size)
    return var_intensity

def var_sqr(img_name, crop_size=None):
    img = cv2.imread(img_name,0)
    if crop_size is not None:
        img = _center_crop(img, crop_size)
    mean = _mean_intensity(img)
    var_intensity = (img - mean)**2
    var_intensity = int(np.sum(var_intensity) / img.size)
    return var_intensity  

def grd_abs(img_name, crop_size=None):
    img = cv2.imread(img_name,0)
    if crop_size is not None:
        img = _center_crop(img, crop_size)
    diff = np.diff(img)
    grd_intensity = np.sum(np.abs(diff))
    return grd_intensity

def grd_sqr(img_name, crop_size=None):
    img = cv2.imread(img_name,0)
    if crop_size is not None:
        img = _center_crop(img, crop_size)
    diff = np.diff(img)
    grd_intensity = np.sum(diff**2)
    return grd_intensity

def correlation(img_name, crop_size=None):
    img = cv2.imread(img_name,0)
    if crop_size is not None:
        img = _center_crop(img, crop_size)
    img1, img2, img3, img4 = img[:-1,:], img[1:,:], img[:-2,:], img[2:,:]
    cor1 = np.sum(np.multiply(img1, img2, dtype=np.int32))
    cor2 = np.sum(np.multiply(img3, img4, dtype=np.int32))
    return cor1-cor2

def e2ntropy(img_name, crop_size=None):
    img = cv2.imread(img_name,0)
    if crop_size is not None:
        img = _center_crop(img, crop_size)
    ent_intensity = [int(pixel*math.log(pixel,2)) for row in img for pixel in row if int(pixel) is not 0]
    ent_intensity = -sum(ent_intensity)
    return ent_intensity 

def entropy(img_name, crop_size=None):
    img = cv2.imread(img_name,0)
    if crop_size is not None:
        img = _center_crop(img, crop_size)
    ent_intensity = img * np.ma.log2(img).filled(0) 
    ent_intensity = -np.sum(ent_intensity, dtype=np.int32)
    return ent_intensity 
