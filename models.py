import utils
import torch
import torch.nn as nn
import pytorch_utils
import torch.optim as optim
import torch.optim as optim
import torch.utils.data as DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

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
    return np.array(new_X), np.array(new_Y)


def loadData(X, Y, num_train, N, batch_size):
    X_train = X[:num_train]
    Y_train = Y[:num_train]
    X_val = X[num_train:]
    Y_val = Y[num_train:]
    print(num_train, N)
    train = DataLoader.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    loader_train = DataLoader.DataLoader(dataset=train,
        batch_size = batch_size,
        sampler=SubsetRandomSampler(range(num_train)))
    val = DataLoader.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))
    loader_val = DataLoader.DataLoader(dataset=val,
                               batch_size = batch_size,
                               sampler=SubsetRandomSampler(range(num_train, N)))
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
    
    X, Y = utils.load_data('mini_data/compressed_256')
    X = X.astype(int)
    Y = Y.astype(int)
    N = X.shape[0]
    num_train = int(N * training_portion)

    # previously, X is: N x 256 x 256 x 3 ; make channels second
    X = np.transpose(X, (0, 3, 1, 2))  # N x 3 x 256 x 256

    X, Y = convertLabelsToClasses(X, Y, N, num_classes)

    loader_train, loader_val = loadData(X, Y, num_train, N, batch_size)
    
    # change this line to try out different models
    model = runTwoLayerCNN(num_classes)
    #model = runFC(hidden_layer_size, num_classes)
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    pytorch_utils.train(model, optimizer, loader_train, loader_val)

if __name__ == '__main__':
  main()