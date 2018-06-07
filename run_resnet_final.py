import utils as local_utils
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim as optim
import torch.utils.data as DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import sys
import matplotlib.pyplot as plt
from PIL import Image
import PIL
from importlib import reload
import torch.nn.functional as F
from torchvision import transforms, utils, models
import os, sys
from random import shuffle

USE_GPU = True
IMG_SZ = 224

dtype = torch.float32

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('using device:', device)

#######################################################################
############## HELPER FUNCTIONS #######################################
#######################################################################

class LandmarksDataset(DataLoader.Dataset):

    def __init__(self, src_folder, transform=None):
        """
        Args:
            src_folder (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.filenames = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f)) 
                          and f != '.DS_Store']
        self.src_folder = src_folder
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.src_folder, self.filenames[idx])
        
        x = Image.open(img_name)

        if self.transform:
            x = self.transform(x)

        imgf = self.filenames[idx]
        y = int(imgf[imgf.index('_') + 1 : imgf.index('.')]) # filename format: [id_label.jpg]
        sample = (x, y)
        return sample
    
def get_loader(directory, batch_size, img_sz=None):
    '''
    takes in directory for train and val data, and returns loaders for both
    applies normalization:
      1. convert values to range 0-1
      2. set mean, std to those specified in pytorch pretrained models (https://pytorch.org/docs/master/torchvision/models.html)
    
    usage:
        loader_train = get_loader(train_directory, batch_sz)
        loader_val = get_loader(val_directory, batch_sz)
    '''
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    if img_sz == None:
        preprocess = transforms.Compose([
            transforms.ToTensor(),  # converts to range 0-1
            normalize               # sets mean, std
        ])
    else:
        preprocess = transforms.Compose([
            transforms.Resize((IMG_SZ, IMG_SZ)), # resize img (ie, for Inception)
            transforms.ToTensor(),  # converts to range 0-1
            normalize               # sets mean, std
        ])
    
    dset = LandmarksDataset(directory, transform=preprocess)
    loader = DataLoader.DataLoader(dataset=dset, batch_size=batch_size)
    
    print ('dataset size', len(dset))
    return loader

def check_accuracy(loader, model): 
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for t, (x, y) in enumerate(loader):
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)

        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        return acc
    
def train(model, optimizer, loader_train, epochs=1, stop=1):
    """
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    iters = []
    losses = []
    for e in range(epochs):
        print ('epoch', e)
        
        num_iters = len(loader_train)
        want_print = 10
        print_every = 50
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
        
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0:
                print(' Iteration %d out of %d, loss = %.4f' % (t, num_iters, loss.item()))
                iters.append(t)
                losses.append(loss.item())
            
            # break early if we only want to use a part of the dataset (for hyperparameter tuning)
            if t > stop * num_iters:
                break

    return iters, losses


#######################################################################################
######## Start running ################################################################
#######################################################################################

print ('training best model')

# use best_lr, best_batch_size
lr = 0.001
batch_size = 20

train_directory = '../data/data_200c/train'
val_directory = '../data/data_200c/val'
test_directory = '../data/data_200c/test'
stats_filename = 'resnet_final_losses.txt'  # for storing iters & losses
saved_model_name = 'best_resnet_model.pt'

# for testing
# train_directory = val_directory = test_directory = '../data/data_200c/mini'
# stats_filename = 'resnet_final_losses_test.txt'  # for storing iters & losses
# saved_model_name = 'best_resnet_model_test.pt'

momentum = 0.9
num_classes = 200

# load data
loader_val = get_loader(val_directory, batch_size)
loader_train = get_loader(train_directory, batch_size)
loader_test = get_loader(test_directory, batch_size)

# set up resnet model with custom final FC layer to predict our number of classes
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Observe that all parameters are being optimized
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

# run for full epoch
num_epochs = 2
iters, losses = train(model, optimizer, loader_train, num_epochs)

# get final accuracies
print (' ')
print ('==========================================================================')
print ('final accuracies')
print ('validation accuracy is ', check_accuracy(loader_val, model))
print ('test accuracy is ', check_accuracy(loader_test, model))
print ('training accuracy is ', check_accuracy(loader_train, model))

# print iters, losses for plotting later
print (' ')
print ('==========================================================================')
print ('iters')
print (iters)
print (' ')
print ('losses')
print (losses)
print (' ')
print ('==========================================================================')

# write iters, losses to file to save
print ('writing losses to file...')
stats_f = open(stats_filename, 'w')
stats_f.write('iters\n')
stats_f.write(str(iters))
stats_f.write('\n\nlosses\n')
stats_f.write(str(losses))
stats_f.write('\n')

# save best model so can be loaded later for saliency maps
print (' ')
print ('saving model')
torch.save(model, saved_model_name)
print ('done!')
