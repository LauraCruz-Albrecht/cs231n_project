import utils
import torch
import torch.nn as nn
import pytorch_utils
import torch.optim as optim
import torch.optim as optim
import torch.utils.data as DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import sys

IMG_SZ = 256

# Takes a range of classes and compresses them from 0 to num_classes - 1. Disregards any
# extra classes.
def convertLabelsToClasses(X, Y, N, num_classes):
    new_X = []
    new_Y = []
    class_num = 0
    id_to_class = {}
    for i in range(0, N):
        if Y[i] in id_to_class:
            new_Y.append(id_to_class[Y[i]])
            new_X.append(X[i])
        else:
            if class_num < num_classes:
                id_to_class[Y[i]] = class_num
                new_Y.append(id_to_class[Y[i]])
                new_X.append(X[i])
                class_num = class_num + 1
    if class_num < num_classes:
        num_classes = class_num
    return np.array(new_X), np.array(new_Y), num_classes


def loadData(X, Y, num_train, N, batch_size):
    X_train = X[:num_train]
    Y_train = Y[:num_train]
    X_val = X[num_train:]
    Y_val = Y[num_train:]
    train = DataLoader.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    loader_train = DataLoader.DataLoader(dataset=train,
        batch_size = batch_size)
    val = DataLoader.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))
    loader_val = DataLoader.DataLoader(dataset=val,
                               batch_size = batch_size)
    return loader_train, loader_val

def runFC(hidden_layer_size, num_classes):
    return pytorch_utils.TwoLayerFC(3 * IMG_SZ * IMG_SZ, hidden_layer_size, num_classes)

def runTwoLayerCNN(num_classes):
    num_channels = 10
    return pytorch_utils.TwoLayerConvNet(3, num_channels, num_classes, 5, 2)

def main():
    batch_size = 64
    num_classes = 500
    hidden_layer_size = 1000
    learning_rate = 1e-2
    training_portion = 0.8
    num_epochs = 10

    directory = sys.argv[1]
    X, Y = utils.load_data(directory)
    X = X.astype(int)
    Y = Y.astype(int)
    N = X.shape[0]

    # previously, X is: N x 256 x 256 x 3 ; make channels second
    X = np.transpose(X, (0, 3, 1, 2))  # N x 3 x 256 x 256
    # X, Y, num_classes = convertLabelsToClasses(X, Y, N, num_classes)
    N = X.shape[0] # need this line because X may have changed in size
    num_train = int(N * training_portion)

    loader_train, loader_val = loadData(X, Y, num_train, N, batch_size)
    
    print("Num classes is ", num_classes)
    print("Num samples being cosidered in training is ", num_train)
    print("Num samples in val is ", N - num_train)

    model = None
    if sys.argv[2] == "2cnn":
        model = runTwoLayerCNN(num_classes)
        print("Running two layer CNN")
    else:
        model = runFC(hidden_layer_size, num_classes)
        print("Running fully connected layer")
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    pytorch_utils.train(model, optimizer, loader_train, loader_val, num_epochs)

if __name__ == '__main__':
  main()
