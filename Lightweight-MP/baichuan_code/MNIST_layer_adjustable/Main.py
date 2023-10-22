import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import Train
import Evaluation

from sklearn.model_selection import train_test_split

import tools
import os

import numpy as np

print('MNIST_Final_layer')
layers = [784, 2000,2000,2000,2000]
length_network = len(layers)-1
print('layers: ' + str(length_network))
Train_flag = True  # True False

# load data
def MNIST_loaders(train_batch_size=60000, test_batch_size=10000):
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=False)  # True

    test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader


def overlay_y_on_x(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_


train_loader, test_loader = MNIST_loaders()

# train data
inputs, targets = next(iter(train_loader))
#inputs, targets = inputs.cuda(), targets.cuda()

# create a validation set
X_train, X_val, y_train, y_val = train_test_split(inputs, targets, test_size=10000, random_state=0)


# train
if Train_flag:

    x_pos = overlay_y_on_x(X_train, y_train)
    y_neg = y_train.clone()
    for idx, y_samp in enumerate(y_train):
        allowed_indices = [0, 1, 2, 3, 4,5,6,7,8,9]
        allowed_indices.pop(y_samp.item())
        y_neg[idx] = torch.tensor(np.random.choice(allowed_indices))
    x_neg = overlay_y_on_x(X_train, y_neg)


    model = Train.build_model(x_pos=x_pos, x_neg=x_neg,layers = layers)

# Load the trained model from saved file
# '4L_2kN_100E_500B' '2L_500N_100E_5000B' '4L_2000N_500E_5kB_50kS' '2L_500N_100E_5kB_50kS'
#name = 'temp'
#model = torch.load(os.path.split(os.path.realpath(__file__))[0]+'/model/' + name)

# evaluation
Evaluation.eval_train_set(model, inputs=X_train, targets=y_train)

# test data
x_te, y_te = next(iter(test_loader))
#x_te, y_te = x_te.cuda(), y_te.cuda()

Evaluation.eval_test_set(model, inputs=x_te, targets=y_te) 

# validation data
Evaluation.eval_val_set(model, inputs=X_val, targets=y_val)

# analysis of validation data
#tools.analysis_val_set_2l(model, inputs=X_val, targets=y_val)
mean,std = tools.analysis_val_set(model, inputs=X_val, targets=y_val,length_network= length_network) # measure each layer's error/ based on the whole sample

Evaluation.eval_val_set_light(model, inputs=x_te, targets=y_te,means = mean, stds = std,length_network= length_network) # measure how many layer needed/ based on each sample




