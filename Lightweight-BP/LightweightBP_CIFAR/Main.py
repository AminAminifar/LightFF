import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader

import Train
import Evaluation

from sklearn.model_selection import train_test_split

import tools
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU for computation.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU for computation.")

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



train_loader, test_loader = CIFAR10_loaders()

# train data
inputs, targets = next(iter(train_loader))
inputs, targets = inputs.to(device), targets.to(device)

# create a validation set
X_train, X_val, y_train, y_val = train_test_split(inputs, targets, test_size=5000, random_state=0)

Train_flag = True  # True False

# train
if Train_flag:
    Train.build_model(x=inputs, targets=targets)

# Load the trained model from saved file
# '4L_2kN_100E_500B' '2L_500N_100E_5000B' '4L_2000N_500E_5kB_50kS' '2L_500N_100E_5kB_50kS'
name = 'LFFmodel'
model = torch.load('model/' + name)

# evaluation
Evaluation.eval_train_set(model, inputs=X_train, targets=y_train)

# test data
x_te, y_te = next(iter(test_loader))
x_te, y_te = x_te.to(device), y_te.to(device)

Evaluation.eval_test_set(model, inputs=x_te, targets=y_te)

# validation data
Evaluation.eval_val_set(model, inputs=X_val, targets=y_val)

# analysis of validation data

confidence_mean_vec, confidence_std_vec = tools.analysis_val_set(model, inputs=X_val, targets=y_val)
print(">>>", confidence_mean_vec, confidence_std_vec)

# Evaluation.eval_val_set_light(model, inputs=X_val, targets=y_val)
# Evaluation.eval_val_set_light(model, inputs=x_te, targets=y_te)  ## temporary use

Evaluation.eval_val_set_light(model, inputs=x_te, targets=y_te,
                              confidence_mean_vec=confidence_mean_vec,
                              confidence_std_vec=confidence_std_vec)

