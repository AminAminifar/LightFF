import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader

import Train
import Evaluation
import numpy as np
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
def CHBMIT_loaders():
    import chbmit_dataset.load_data as chb

    x_train= np.zeros((1,32,32))
    x_test= np.zeros((1,32,32))
    y_train = np.zeros(1)
    y_test = np.zeros(1)

    chb_list = range(1,25,1)
    acc_total = 0
    #print("-------------begin-------------------")
    for i in chb_list:
        #print("process for patient "+str(i))
        if i== 6 or i==14 or i==16: # we do not consider patient 6/14/16
            continue
        if i <10:
            which_patients = 'chb0'+str(i)
        else:
            which_patients = 'chb'+str(i)

        X_train_example,Y_train_example,X_val_example,Y_val_example,X_test_example,Y_test_example = chb.load_data(which_patients)
        #X_train_example = np.power(X_train_example[:,::2,:],2)
        #X_val_example = np.power(X_val_example[:,::2,:],2)
        #X_test_example = np.power(X_test_example[:,::2,:],2)

        X_train_example = X_train_example[:,::2,:]
        X_val_example = X_val_example[:,::2,:]
        X_test_example = X_test_example[:,::2,:]

        x_train=np.vstack((x_train,X_train_example.reshape(X_train_example.shape[0],32,32)))
        x_train=np.vstack((x_train,X_val_example.reshape(X_val_example.shape[0],32,32)))
        x_test=np.vstack((x_test,X_test_example.reshape(X_test_example.shape[0],32,32)))

        y_train = np.hstack((y_train,Y_train_example))
        y_train = np.hstack((y_train,Y_val_example))
        y_test = np.hstack((y_test,Y_test_example))

    x_train = x_train[1:,:,:]
    x_test = x_test[1:,:,:]
    y_train = y_train[1:]
    y_test = y_test[1:]

    x_train=x_train.reshape(x_train.shape[0], -1)
    x_test=x_test.reshape(x_test.shape[0], -1)

    train_data = torch.utils.data.TensorDataset(torch.tensor(x_train).float(),torch.tensor(y_train).long())
    test_data = torch.utils.data.TensorDataset(torch.tensor(x_test).float(),torch.tensor(y_test).long())

    train_loader = torch.utils.data.DataLoader(train_data,batch_size=5091,shuffle=True,pin_memory=True,num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=530,shuffle=True,pin_memory=True,num_workers=0)
    return train_loader,test_loader



train_loader, test_loader = CHBMIT_loaders()

# train data
inputs, targets = next(iter(train_loader))
inputs, targets = inputs.to(device), targets.to(device)

# create a validation set
X_train, X_val, y_train, y_val = train_test_split(inputs, targets, test_size=1000, random_state=0)

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

