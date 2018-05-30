import csv
import numpy as np
import pandas as pd
import os, sys
from PIL import Image
import numpy as np
from time import time
from time import sleep

# global variables
IMG_SZ = 256    # width, height of resized image square

def load_image(infilename):
    '''
    opens image at provided path, returns numpy array 
    https://stackoverflow.com/questions/7762948/how-to-convert-an-rgb-image-to-numpy-array
    '''
    img = Image.open(infilename)
    img.load()
    data = np.asarray( img, dtype="int32" ) # should this be 'uint8' ??
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

def load_data(src_folder):
    src_files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f)) and f != '.DS_Store']

    # can't do over 50k files because of google cloud memory restrictions
    estimated_N = len(src_files)
    if len(src_files) > 50000:
        estimated_N = 20000

    X, Y = np.empty([estimated_N, IMG_SZ, IMG_SZ, 3]), np.empty([estimated_N])

    position = 0
    for i in range(len(src_files)):
        _file = src_files[i]
        x = load_image(src_folder + '/' + _file)  # numpy array [IMG_SZ x IMG_SZ x 3]
        y = _file[_file.index('_') + 1 : _file.index('.')]  # filename format: [id_label.jpg]
        
        if int(y) >= 100: continue

        X[position] = x
        Y[position] = y
        position = position + 1

        if i % (1000) == 0: print ('i', i)
        
    X = X[:position]
    Y = Y[:position]
    return X, Y


# to demonstrate loading data
def main():
    src_folder = 'mini_data/compressed_576_s'
    X, Y = load_data(src_folder)
    print (X.shape, Y.shape)

if __name__ == '__main__':
  main()

