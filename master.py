import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import numpy as np
import zipfile
import pandas as pd

with zipfile.ZipFile("train.csv 4.30.14 PM.zip") as z:
    with z.open("train.csv") as f:
        train = pd.read_csv(f, header=0, delimiter=",")
        print(train.head())    # print the first 5 rows
        
        # x_train = df[['id']]
        # y_train = df['landmark_id'].values.astype(np.float32)


        print(type(train))
        print(train.shape)

        images = []
        num_dropped = 0
        for index, row in train.iterrows():
            id_num, url, landmark_id = row
            try:
                with open('mini_data/images_10/' + id_num + '.jpg') as f:
                    print(type(f))
                    images = images.append(f.read())
            except:
                num_dropped = num_dropped + 1
                train.drop(index, inplace=True)
        print(num_dropped)