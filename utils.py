import csv
import numpy as np
import pandas as pd
import os, sys
from PIL import Image
import numpy as np
from time import time
from time import sleep

# global variables
IMG_SZ = 576    # width, height of resized image square

def load_image(infilename):
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

def load_data(src_folder):
    X, Y = [], []

    src_files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f)) and f != '.DS_Store']

    N = len(src_files)

    for i in range(N):
        _file = src_files[i]
        x = load_image(src_folder + '/' + _file)  # numpy array [IMG_SZ x IMG_SZ x 3]
        y = _file[_file.index('_') + 1 : _file.index('.')]  # filename format: [id_label.jpg]
        
        X.append(x)
        Y.append(y)

        if i % (1000) == 0: print ('i', i)
        
        # can't do over 50k files because of google cloud restrictions
        # if i == 50000:
        #     break

    X = np.array(X)
    Y = np.array(Y)
    return X, Y


# to demonstrate loading data
def main():
    src_folder = 'mini_data/compressed_576_s'
    X, Y = load_data(src_folder)
    print (X.shape, Y.shape)

if __name__ == '__main__':
  main()

