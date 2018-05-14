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
from utils import *  # our helper functions

'''
Takes directory of unprocessed images, and crops/rescales each image and
puts it in a new directory
'''

IMG_SZ = 576    # width, height of resized image square

# specify src folder and dest folder. both must exist (though latter is empty)
src_folder = 'mini_data/images_1000'
dest_folder = 'mini_data/compressed_%d/' % IMG_SZ

def crop_and_resize():
    '''
    reads through folder of raw images, crops each image to square, 
    resizes to standard size, then saves to output folder
    '''

    src_files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f)) and f != '.DS_Store']
    src_files = src_files[:10]

    print('Working with {0} images'.format(len(src_files)))

    N = len(src_files)
    for i in range(N):
        _file = src_files[i]
        img_arr = load_image(src_folder + '/' + _file)
        img_arr_cropped = crop_image(img_arr)
        img_arr_rescaled = scipy.misc.imresize(img_arr_cropped, (IMG_SZ, IMG_SZ, 3))
        img_final = Image.fromarray(img_arr_rescaled)
        img_final.save(dest_folder + _file)
        if i % 50 == 0: print ('num processed:', i)

crop_and_resize()
