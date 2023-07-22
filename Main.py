import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader

import Train
import Evaluation


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
        batch_size=train_batch_size, shuffle=True)

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
inputs, targets = inputs.cuda(), targets.cuda()

Train_flag = False

# train
if Train_flag:
    x_pos = overlay_y_on_x(inputs, targets)
    rnd = torch.randperm(inputs.size(0))
    x_neg = overlay_y_on_x(inputs, targets[rnd])

    Train.build_model(x_pos=x_pos, x_neg=x_neg)

# Load the trained model from saved file
name = '2L_500N_100E_5000B'  # '4L_2kN_100E_500B' '2L_500N_100E_5000B'
model = torch.load('model/' + name)

# evaluation
Evaluation.eval_train_set(model, inputs=inputs, targets=targets)

# test data
x_te, y_te = next(iter(test_loader))
x_te, y_te = x_te.cuda(), y_te.cuda()

Evaluation.eval_test_set(model, inputs=x_te, targets=y_te)


