# -*- coding: utf-8 -*-

# !/usr/bin/python

# Note to Kagglers: This script will not run directly in Kaggle kernels. You
# need to download it and run it on your local machine.

# Downloads images from the Google Landmarks dataset using multiple threads.
# Images that already exist will not be downloaded again, so the script can
# resume a partially completed download. All images will be saved in the JPG
# format with 90% compression quality.

# https://www.kaggle.com/maxwell110/python3-version-image-downloader

import sys, os, multiprocessing, csv
from urllib import request, error
from PIL import Image
from io import BytesIO

import numpy as np
import scipy.misc

IMG_SZ = 256  # width, height of resized image square

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

def parse_data(data_file):
    csvfile = open(data_file, 'r')
    csvreader = csv.reader(csvfile)
    #key_url_list = [line[:2] for line in csvreader]
    key_url_list = [line for line in csvreader]
    #return key_url_list[1:]  # Chop off header
    return key_url_list


def download_image(key_url):
    out_dir = sys.argv[2]
    # (key, url, label) = key_url
    # filename = os.path.join(out_dir, '{}_{}.jpg'.format(key, label))

    # modified data
    (key, url, old_label, new_label) = key_url
    filename = os.path.join(out_dir, '{}_{}.jpg'.format(key, new_label))

    if os.path.exists(filename):
        print('Image {} already exists. Skipping download.'.format(filename))
        return

    try:
        response = request.urlopen(url)
        image_data = response.read()
    except:
        print('Warning: Could not download image {} from {}'.format(key, url))
        return

    try:
        pil_image = Image.open(BytesIO(image_data))
    except:
        print('Warning: Failed to parse image {}'.format(key))
        return

    try:
        pil_image_rgb = pil_image.convert('RGB')
    except:
        print('Warning: Failed to convert image {} to RGB'.format(key))
        return

    try:
      img_arr = np.asarray(pil_image_rgb, dtype="int32" )
      img_arr_cropped = crop_image(img_arr)
      img_arr_rescaled = scipy.misc.imresize(img_arr_cropped, (IMG_SZ, IMG_SZ, 3))
      img_final = Image.fromarray(img_arr_rescaled)
    except:
      print('Warning: Failed to crop/resize image %s' % key)
      return
    
    try:
      img_final.save(filename, format='JPEG', quality=90)
    except:
      print('Warning: Failed to save image %s' % filename)
      return


def loader():
    if len(sys.argv) != 3:
        print('Syntax: {} <data_file.csv> <output_dir/>'.format(sys.argv[0]))
        sys.exit(0)
    (data_file, out_dir) = sys.argv[1:]

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    key_url_list = parse_data(data_file)
    pool = multiprocessing.Pool(processes=4)  # Num of CPUs
    pool.map(download_image, key_url_list)
    pool.close()
    pool.terminate()


# arg1 : data_file.csv
# arg2 : output_dir
if __name__ == '__main__':
    loader()