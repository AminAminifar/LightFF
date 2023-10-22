import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader

import Train
import Evaluation

from sklearn.model_selection import train_test_split

import tools

import torchvision
import torchvision.transforms as transforms


# load data
def CIFAR10_loaders(train_batch_size=50000, test_batch_size=10000):
    transform = Compose([transforms.ToTensor(),
                         Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        Lambda(lambda x: torch.flatten(x))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                               shuffle=False)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                              shuffle=False)

    return train_loader, test_loader


def overlay_y_on_x(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    x_ = x.clone()
    ch1_i = 0
    ch2_i = 1024
    ch3_i = 2048
    x_[:, ch1_i:ch1_i + 10] *= 0.0
    x_[:, ch2_i:ch2_i + 10] *= 0.0
    x_[:, ch3_i:ch3_i + 10] *= 0.0
    x_[range(x.shape[0]), ch1_i + y] = x.max()
    x_[range(x.shape[0]), ch2_i + y] = x.max()
    x_[range(x.shape[0]), ch3_i + y] = x.max()
    return x_


train_loader, test_loader = CIFAR10_loaders()

# train data
dataiter = iter(train_loader)
inputs, targets = next(dataiter)
inputs, targets = inputs.cuda(), targets.cuda()

# create a validation set
X_train, X_val, y_train, y_val = train_test_split(inputs, targets, test_size=5000, random_state=0)
#
Train_flag = False  # True False

# train
if Train_flag:
    x_pos = overlay_y_on_x(X_train, y_train)  # inputs, targets
    rnd = torch.randperm(X_train.size(0))  # inputs.size(0)
    x_neg = overlay_y_on_x(X_train, y_train[rnd])  # inputs, targets[rnd]

    Train.build_model(x_pos=x_pos, x_neg=x_neg)

# Load the trained model from saved file
# '4L_2kN_100E_500B' '2L_500N_100E_5000B' '4L_2000N_500E_5kB_50kS' '2L_500N_100E_5kB_50kS'
name = 'temp'
model = torch.load('model/' + name)

# evaluation
Evaluation.eval_train_set(model, inputs=X_train, targets=y_train)

# test data
x_te, y_te = next(iter(test_loader))
x_te, y_te = x_te.cuda(), y_te.cuda()

Evaluation.eval_test_set(model, inputs=x_te, targets=y_te)

# validation data
Evaluation.eval_val_set(model, inputs=X_val, targets=y_val)

# analysis of validation data
# tools.analysis_val_set_2l(model, inputs=X_val, targets=y_val)
# tools.analysis_val_set(model, inputs=X_val, targets=y_val)

Evaluation.eval_val_set_light(model, inputs=X_val, targets=y_val)
# Evaluation.eval_val_set_light(model, inputs=x_te, targets=y_te)  ## temporary use


