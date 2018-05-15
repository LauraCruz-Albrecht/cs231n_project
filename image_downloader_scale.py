#!/usr/bin/python

# Note to Kagglers: This script will not run directly in Kaggle kernels. You
# need to download it and run it on your local machine.

# Downloads images from the Google Landmarks dataset using multiple threads.
# Images that already exist will not be downloaded again, so the script can
# resume a partially completed download. All images will be saved in the JPG
# format with 90% compression quality.

# https://www.kaggle.com/tobwey/landmark-recognition-challenge-image-downloader
# note: run using python2
# NOTE: assumes running on train data

'''
Note: this version will also crop/resize each image before saving
'''

import sys, os, multiprocessing, csv
import urllib2

from PIL import Image
from StringIO import StringIO

import numpy as np
import scipy.misc

#IMG_SZ = 576    # width, height of resized image square
IMG_SZ = 256

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

def ParseData(data_file):
  csvfile = open(data_file, 'r')
  csvreader = csv.reader(csvfile)

  # if training: [key, url, label]
  # if test: [key, url]
  key_url_list = [line for line in csvreader]

  return key_url_list[1:]  # Chop off header?


def DownloadImage(key_url):
  out_dir = sys.argv[2]

  # if train
  (key, url, label) = key_url
  filename = os.path.join(out_dir, '%s_%s.jpg' % (key, label))

  # if test
  # (key, url) = key_url
  # filename = os.path.join(out_dir, '%s.jpg' % key)

  if os.path.exists(filename):
    print('Image %s already exists. Skipping download.' % filename)
    return

  try:
    response = urllib2.urlopen(url)
    image_data = response.read()
  except:
    print('Warning: Could not download image %s from %s' % (key, url))
    return

  try:
    pil_image = Image.open(StringIO(image_data))
  except:
    print('Warning: Failed to parse image %s' % key)
    return

  try:
    pil_image_rgb = pil_image.convert('RGB')
  except:
    print('Warning: Failed to convert image %s to RGB' % key)
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

def Run():
  if len(sys.argv) != 3:
    print('Syntax: %s <data_file.csv> <output_dir/>' % sys.argv[0])
    sys.exit(0)
  (data_file, out_dir) = sys.argv[1:]

  if not os.path.exists(out_dir):
    os.mkdir(out_dir)

  key_url_list = ParseData(data_file)
  pool = multiprocessing.Pool(processes=50)
  pool.map(DownloadImage, key_url_list)


if __name__ == '__main__':
  Run()