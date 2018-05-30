import csv
import numpy as np
import pandas as pd
from past.builtins import xrange
import os, sys
from IPython.display import display
from IPython.display import Image as _Imgdis
from PIL import Image
import numpy as np
from time import time
from time import sleep
from scipy import ndimage
from matplotlib import pyplot as plt
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import scipy
import pickle
import gzip

# global variables
IMG_SZ = 576    # width, height of resized image square

def load_image(infilename) :
    '''
    opens image at provided path, returns numpy array 
    https://stackoverflow.com/questions/7762948/how-to-convert-an-rgb-image-to-numpy-array
    '''
    img = Image.open(infilename)
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

def crop_image(img_arr):
    '''
    takes in np array of image
    returns largest centered square, also as np array
    '''
    h, w, _ = img_arr.shape
    
    if w > h:
        offset = (w - h) // 2
        return img_arr[:, offset : h + offset, :]
    elif h > w:
        offset = (h - w) // 2
        return img_arr[offset : w + offset, :]

    # h == w
    return img_arr

def get_stats():
    '''
    iterates over all raw images in folder, prints average cropped sz
    '''
    sizes = []
    for i in range(N):
        img_filename = folder + "/" + onlyfiles[i]
        img_arr = load_image(img_filename)
        img_arr_cropped = crop_image(img_arr)
        sz = len(img_arr_cropped)

        sz_sum += sz
        sizes.append(sz)

        bucket = sz // 100  # bucket by nearest multiple of 100 (floored)
        if bucket not in hist: hist[bucket] = 1
        else: hist[bucket] += 1

    print ('avg sz', sz_sum / N)
    print ('min sz', np.min(sizes), 'max sz', np.max(sizes))
    print ('avg sz', np.mean(sizes), 'variance', np.var(sizes))

    X = list(hist.keys())
    Y = [hist[k] for k in hist]
    plt.plot(X, Y, linestyle='None', marker='o', markeredgecolor='blue', markerfacecolor='white')

    plt.xlabel('bucket size')
    plt.ylabel('number of images with this size (cropped)')
    plt.title('images by size')
    plt.legend()
    plt.show()

    # most common sizes
    tups = [(hist[k], k * 100) for k in hist]
    tups.sort(reverse=True)
    print (tups[:10])
    print (np.sum([t[0] for t in tups]))

def demo_crop_resize():
    '''
    demonstrates how cropping / resizing works on images
    '''
    NUM_DEMO = 5  # number of images to demo processing

    src_folder = 'mini_data/images_1000'
    src_files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    src_files = src_files[: NUM_DEMO]

    for _file in src_files:
        img_arr = img_arr = load_image(src_folder + '/' + _file)
        img_arr_cropped = crop_image(img_arr)
        img_arr_rescaled = scipy.misc.imresize(img_arr_cropped, (IMG_SZ, IMG_SZ, 3))

        to_show = [img_arr, img_arr_cropped, img_arr_rescaled]
        for arr in to_show:
            plt.imshow(arr.astype('uint8'))
            plt.show()

# --------------------------------

#get_stats()
demo_crop_resize()

