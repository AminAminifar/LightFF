import torch
import torch.nn as nn

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

import numpy as np

from sklearn.utils import shuffle

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report, precision_score, recall_score

from tqdm import tqdm
import time

if torch.cuda.is_available():
    device = torch.device("cuda")

else:
    device = torch.device("cpu")



class MyModel(nn.Module):
    def my_block(self, in_size, out_size):
        block = torch.nn.Sequential(
            nn.Linear(in_size, out_size),
            # torch.nn.BatchNorm1d(out_size),
            torch.nn.ReLU(),
        )
        return block

    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = self.my_block(in_size=3072, out_size=2000)  # Input size: 28*28 for flattened MNIST images
        self.fc2 = self.my_block(in_size=2000, out_size=2000)
        self.fc3 = self.my_block(in_size=2000, out_size=2000)
        self.fc4 = self.my_block(in_size=2000, out_size=2000)
        
        self.fc6 = self.my_block(in_size=2000, out_size=2000)
        self.fc7 = self.my_block(in_size=2000, out_size=2000)

        self.fc5 = nn.Linear(2000, 10)  # Output size: 10 for 10 classes in MNIST
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        # x = x.view(-1, 28 * 28)  # Flatten the input images
        x = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        x = self.fc1(x)
        x = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        x = self.fc2(x)
        x = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        x = self.fc3(x)
        x = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        x = self.fc4(x)

        x = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        x = self.fc6(x)

        x = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        x = self.fc7(x)

        x = self.fc5(x)
        x = self.sm(x)
        return x


# Instantiate the model
model = MyModel().to(device)


##
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

def train_BPmodel(X,Y):

    train_loader, test_loader = CIFAR10_loaders()
    #
    # # train data
    # inputs, targets = next(iter(train_loader))
    # inputs, targets = inputs.cuda(), targets.cuda()
    inputs, targets = X,Y

    # create a validation set
    # X_train, X_val, y_train, y_val = train_test_split(inputs, targets, test_size=10000, random_state=0)

    # test data
    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.to(device), y_te.to(device)
    #################################################################
    train_data = inputs  # X_train
    train_labels = targets  # y_train
    # define the optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # lr=0.0001

    # enumerate epochs
    for epoch in tqdm(range(100)):

        train_data_record_indices = range(0, train_data.shape[0])
        train_data_record_indices_shuffled = shuffle(train_data_record_indices)

        batch_size = 500  # 64
        num_batches = int(train_data.shape[0] / batch_size)
        chunk_indices = np.array_split(train_data_record_indices_shuffled, num_batches)

        # enumerate mini batches
        for i in range(num_batches):  # range(num_batches)

            inputs, targets = train_data[chunk_indices[i]], train_labels[chunk_indices[i]]
            # inputs = np.reshape(inputs, (inputs.shape[0], 1, 28, 28))
            # inputs = torch.Tensor(inputs)
            # targets = torch.Tensor(targets)
            # targets = targets.type(torch.LongTensor)
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
    # return model
    # torch.save(model.state_dict(), 'model/BP/BPmodel')
    # ##################################
    test_data, test_labels = x_te, y_te

    inputs, targets = test_data, test_labels
    # inputs = np.reshape(inputs, (inputs.shape[0], 1, 28, 28))
    # inputs = torch.Tensor(inputs)
    # targets = torch.Tensor(targets)
    # targets = targets.type(torch.LongTensor)


    # evaluate the model on the test set
    yhat = model(inputs)
    # retrieve numpy array
    yhat = yhat.cpu().detach().numpy()
    actual = targets.cpu().numpy()
    # convert to class labels
    yhat = np.argmax(yhat, axis=1)
    # reshape for stacking
    actual = actual.reshape((len(actual), 1))
    yhat = yhat.reshape((len(yhat), 1))

    # f1_performance = f1_score(actual, yhat, average='weighted')
    acc_performance = accuracy_score(actual, yhat)
    # precision_performance = precision_score(actual, yhat)
    # recall_performance = recall_score(actual, yhat)

    print("\naccuracy_score: ", acc_performance)
    print(sum(a == b for a, b in zip(actual, yhat)))
    time.sleep(.1)
    #############################
    return model

