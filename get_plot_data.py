#!/usr/bin/python
import sys, os, multiprocessing, csv
import numpy as np
from random import shuffle

'''
takes a file that is the output of a training run, ie:

 Iteration 0 out of 3040, loss = 5.6261
 Iteration 50 out of 3040, loss = 5.4841
 Iteration 100 out of 3040, loss = 5.2629
 Iteration 150 out of 3040, loss = 5.2399
 ...
 
and prints out corresponding losses and iters lists that can
be used for plotting.
'''

def get_data(f):
  iters, losses = [], []
  while True:
    line = f.readline().strip()
    if 'batch' in line:
      break

    toks = line.split()
    iters.append(int(toks[1]))
    losses.append(float(toks[-1]))

  # this line has lr, bs
  toks = line.split()

  lr = float(toks[-1].strip())
  batch_size = int(toks[-3].strip())
  s = 'lr: %f, batch sz: %d' % (lr, batch_size)

  f.readline()  # skip blank line
  return (iters, losses, s)

#FILENAME = 'resnet_hyperparam.txt'
#FILENAME = 'vgg_hyperparam.txt'
FILENAME = 'DL_hyperparam.txt'

f = open(FILENAME, 'r')
plotting_data = []
for i in range(6):
  data = get_data(f)
  plotting_data.append(data)
f.close()

# check data
# print (plotting_data)
# print (' ')
N = len(plotting_data)
print ('[')
for i in range(N-1):
  print (plotting_data[i], ',')
print (plotting_data[N-1])
print (']')