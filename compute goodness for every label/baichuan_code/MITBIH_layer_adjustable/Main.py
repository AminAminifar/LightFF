import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader

import Train
import Evaluation
from sklearn.model_selection import train_test_split
import tools
import os
import matplotlib.pyplot as plt
import mitbih_dataset.load_data as bih 
import numpy as np

print('MITBIH_Final_layer')
layers = [187, 2000,2000,2000,2000]
length_network = len(layers)-1
print('layers: ' + str(length_network))
Train_flag = True  # True False

def overlay_y_on_x(x, y):
    """Replace the first 5 pixels of data [x] with one-hot-encoded label [y]
    """
    x_ = x.clone()
    x_[:, :5] *= 0.0
    y=y.to(torch.int64)
    x_[range(x.shape[0]), y] = x.max()
    return x_

X_train, Y_train, X_test, Y_test = bih.load_data(1)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=10000, random_state=0)


X_val = X_val.squeeze(2)
X_train = X_train.squeeze(2)
X_test  = X_test.squeeze(2)


X_train = torch.tensor(X_train)
Y_train = torch.tensor(Y_train)
X_test  = torch.tensor(X_test)
Y_test  = torch.tensor(Y_test)
X_val = torch.tensor(X_val)
Y_val = torch.tensor(Y_val)

# train
if Train_flag:
    x_pos = overlay_y_on_x(X_train, Y_train)
    y_neg = Y_train.clone()
    for idx, y_samp in enumerate(Y_train):
        allowed_indices = [0, 1, 2, 3, 4]
        allowed_indices.pop(y_samp.item())
        y_neg[idx] = torch.tensor(np.random.choice(allowed_indices))
    x_neg = overlay_y_on_x(X_train, y_neg)

    model=Train.build_model(x_pos=x_pos, x_neg=x_neg,layers = layers)

# Load the trained model from saved file
# name = 'temp'   #'4L_2kN_100E_500B' '2L_500N_100E_5000B'
# model = torch.load(os.path.split(os.path.realpath(__file__))[0]+'/model/' + name)

# evaluation
Evaluation.eval_train_set(model, inputs=X_train, targets=Y_train)

# test data

Evaluation.eval_test_set(model, inputs=X_test, targets=Y_test)
Evaluation.eval_val_set(model, inputs=X_val, targets=Y_val)

mean,std = tools.analysis_val_set(model, inputs=X_val, targets=Y_val,length_network= length_network)
Evaluation.eval_val_set_light(model, inputs=X_test, targets=Y_test,means = mean, stds = std,length_network= length_network)

