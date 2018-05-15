import utils
import torch
import torch.nn as nn
import pytorch_utils
import torch.optim as optim
import torch.optim as optim
import torch.utils.data as DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


# Takes a range of classes and compressed them from 0 to num_classes - 1
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
    train = DataLoader.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    loader_train = DataLoader.DataLoader(dataset=train,
        batch_size = batch_size,
        sampler=SubsetRandomSampler(range(num_train)))
    val = DataLoader.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))
    loader_val = DataLoader.DataLoader(dataset=val,
                               batch_size = batch_size,
                               sampler=SubsetRandomSampler(range(num_train, N)))
    return loader_train, loader_val

def runFC():
    return pytorch_utils.TwoLayerFC(3 * 256 * 256, hidden_layer_size, num_classes)

# def runTwoLayerCNN():
    # return pytorch_utils.TwoLayerFC(3 * 256 * 256, hidden_layer_size, num_classes)

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

    X, Y = convertLabelsToClasses(X, Y, N, num_classes)
    loader_train, loader_val = loadData(X, Y, num_train, N, batch_size)
    model = runFC(hidden_layer_size, num_classes)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    pytorch_utils.train(model, optimizer, loader_train, loader_val)

if __name__ == '__main__':
  main()