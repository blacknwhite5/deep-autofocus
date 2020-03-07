"""
@author H. Kim, M. Oh
@company EgoVid Inc
"""
import numpy as np
import math
import cv2

__all__ = ['var_abs', 'var_sqr', 'grd_abs', 'grd_sqr', 'correlation', 'entropy']

def _mean_intensity(img):
    return np.sum(img, dtype=np.int32) / img.size

def CenterCrop(sharpness_fn):
    def wrapper(*args, **kwargs):
        img = kwargs['img'] if 'img' in kwargs.keys() else args[0]
        img = cv2.imread(img,0)
        if len(args) == 2 or 'crop_size' in kwargs.keys():
            crop_size = kwargs['crop_size'] if 'crop_size' in kwargs.keys() else args[1]
            h, w = img.shape
            cntr_h, cntr_w = h//2, w//2
            crop_h, crop_w = crop_size[0]//2, crop_size[1]//2
            img = img[cntr_h-crop_h:cntr_h+crop_h, cntr_w-crop_w:cntr_w+crop_w]
        return sharpness_fn(img)
    return wrapper

@CenterCrop
def var_abs(img):
    mean = _mean_intensity(img)
    var_intensity = np.abs(img - mean)
    var_intensity = int(np.sum(var_intensity) / img.size)
    return var_intensity

@CenterCrop
def var_sqr(img):
    mean = _mean_intensity(img)
    var_intensity = (img - mean)**2
    var_intensity = int(np.sum(var_intensity) / img.size)
    return var_intensity  

@CenterCrop
def grd_abs(img):
    diff = np.diff(img)
    grd_intensity = np.sum(np.abs(diff))
    return grd_intensity

@CenterCrop
def grd_sqr(img):
    diff = np.diff(img)
    grd_intensity = np.sum(diff**2)
    return grd_intensity

@CenterCrop
def correlation(img):
    img1, img2, img3, img4 = img[:-1,:], img[1:,:], img[:-2,:], img[2:,:]
    cor1 = np.sum(np.multiply(img1, img2, dtype=np.int32))
    cor2 = np.sum(np.multiply(img3, img4, dtype=np.int32))
    return cor1-cor2

@CenterCrop
def entropy(img):
    ent_intensity = img * np.ma.log2(img).filled(0) 
    ent_intensity = -np.sum(ent_intensity, dtype=np.int32)
    return ent_intensity 
