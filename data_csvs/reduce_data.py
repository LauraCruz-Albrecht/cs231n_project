#!/usr/bin/python

'''
This takes in a list of the top 500 classes and reduces into C classes
nd N examples per class, split 80/20 into a train and test csv
'''


import sys, os, multiprocessing, csv
import numpy as np
from random import shuffle

'''
def main(): # 50 x 20 = 1000
  top_classes_file = 'top_500_classes.txt'
  data_file = 'train.csv'

  out_prefix = '100c_1000i'

  num_classes = 100
  num_per_class = 1000

  top_classes = np.loadtxt(top_classes_file).tolist()[:num_classes]
  csvfile = open(data_file, 'r')

  outfile = open(out_file, 'w', newline='')

  counter = [0] * num_classes

  csvfile.readline() # skip header
  for line in csvfile:
    c = int(line.split(',')[-1])

    if c in top_classes:
      c_mapped = top_classes.index(c)

      if counter[c_mapped] < num_per_class:
        #print (c, c_mapped)
        out_line = line.strip() + ',' + str(c_mapped) + '\n'
        outfile.write(out_line)
        counter[c_mapped] += 1

    # check if can break early
    if sum(counter) == num_classes * num_per_class:
      print ('breaking early')
      break

  csvfile.close()
  outfile.close()
'''

def v1(): 
  top_classes_file = 'top_500_classes.txt'
  data_file = 'train.csv'

  out_prefix = '100c_1000i'

  num_classes = 100
  num_per_class = 1000

  top_classes = np.loadtxt(top_classes_file).tolist()[:num_classes]

  newlines = []  # where we'll store the lines for the reduced csv's
  counter = [0] * num_classes  # histogram to track number read per class so far

  csvfile = open(data_file, 'r')
  csvfile.readline() # skip header of train.txt
  for line in csvfile:
    c = int(line.split(',')[-1])

    if c in top_classes:
      c_mapped = top_classes.index(c)

      if counter[c_mapped] < num_per_class:
        out_line = line.strip() + ',' + str(c_mapped) + '\n'
        newlines.append(out_line)
        counter[c_mapped] += 1

    # check if can break early
    if sum(counter) == num_classes * num_per_class:
      print ('breaking early')
      break

  csvfile.close()

  N_tot = len(newlines)  # should be num_classes x num_per_class
  print ('num new lines', N_tot)
  

  # shuffle newlines and split into train/test
  print ('shuffling...')
  shuffle(newlines)
  N_train = int(N_tot * 0.9)

  trainlines = newlines[: N_train]
  testlines = newlines[N_train :]

  print ('writing to train and test.csv')
  trainfile = open(out_prefix + '_train.csv', 'w', newline='')
  for line in trainlines:
    trainfile.write(line)
  trainfile.close()

  testfile = open(out_prefix + '_test.csv', 'w', newline='')
  for line in testlines:
    testfile.write(line)
  testfile.close()

def v2():
  # get just 400 images per class for the 100 classes
  src_file = '100c_1000i_train.csv'
  dest_file = '100c_400i_train.csv'

  num_classes = 100
  num_per_class = 400

  csvfile = open(src_file, 'r')

  outfile = open(dest_file, 'w', newline='')

  counter = [0] * num_classes

  for line in csvfile:
    c = int(line.split(',')[-1])

    if counter[c] < num_per_class:
      outfile.write(line)
      counter[c] += 1

    # check if can break early
    if sum(counter) == num_classes * num_per_class:
      print ('breaking early')
      break

  csvfile.close()
  outfile.close()

#v1()
v2()