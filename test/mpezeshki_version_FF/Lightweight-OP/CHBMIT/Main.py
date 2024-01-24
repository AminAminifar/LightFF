import torch

import Train
import Evaluation
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tools
import os

import numpy as np
import CHBMIT.chbmit_dataset.load_data as chb

print('CHBMIT_One_Pass')
layers = [1024, 200,200,200,200]
length_network = len(layers)-1
print('layers: ' + str(length_network))
print(layers)
Train_flag = True  # True False

def overlay_y_on_x(x, y):
    """Replace the first 2 pixels of data [x] with one-hot-encoded label [y]
    """
    x_ = x.clone()
    x_[:, :2] *= 0.0
    y=y.to(torch.int64)
    x_[range(x.shape[0]), y] = 41.85#x.max()
    return x_

def overlay_on_x_neutral(x):
    """Replace the first 10 pixels of data [x] with 0.1s
    """
    x_ = x.clone()
    x_[:, :2] *= 0.0
    x_[range(x.shape[0]), :2] = 0.1  # x.max()
    return x_



def each_patient_detection(which_patients,Train_flag,layers,length_network):

    torch.manual_seed(1234)

    X_train,Y_train,X_val,Y_val,X_test,Y_test = chb.load_data(which_patients)

    X_train = X_train.squeeze(2)
    X_val = X_val.squeeze(2)
    X_test = X_test.squeeze(2)

    train_size =X_train.shape[0]
    val_size =X_val.shape[0]
    test_size =X_test.shape[0]

    X_train = X_train[:,::2]
    X_test = X_test[:,::2]
    X_val = X_val[:,::2]


    X_train = torch.tensor(X_train).pow(2) # pow 2 better
    Y_train = torch.tensor(Y_train).to(torch.int64)
    X_val = torch.tensor(X_val).pow(2)

    Y_val = torch.tensor(Y_val).to(torch.int64)
    X_test = torch.tensor(X_test).pow(2)
    Y_test = torch.tensor(Y_test).to(torch.int64)


    # train
    if Train_flag:
        x_pos = overlay_y_on_x(X_train, Y_train)
        x_neg = overlay_y_on_x(X_train, 1-Y_train)
        x_neutral = overlay_on_x_neutral(X_train)
        model = Train.build_model(x_pos=x_pos, x_neg=x_neg, x_neutral=x_neutral, targets=Y_train,layers=layers)

    #name = 'temp'
    #model = torch.load(os.path.split(os.path.realpath(__file__))[0]+'/model/' + name+which_patients)

    train_error = Evaluation.eval_train_set(model, inputs=X_train, targets=Y_train,size = train_size)
    test_error = Evaluation.eval_test_set(model, inputs=X_test, targets=Y_test, size = test_size) 
    val_error = Evaluation.eval_val_set(model, inputs=X_val, targets=Y_val,size = val_size )#val_size)

    mean,std = tools.analysis_val_set(model, inputs=X_val, targets=Y_val,size=val_size)
    confidence_mean_vec = mean
    confidence_std_vec = std

    test_error_light, light_number_layers,percentage = Evaluation.eval_val_set_light(model, inputs=X_test, targets=Y_test, confidence_mean_vec=confidence_mean_vec, confidence_std_vec=confidence_std_vec,size=test_size)  ## temporary use

    return train_error,val_error,test_error,test_error_light,light_number_layers,percentage

def train_seizure(Train_flag,layers,length_network):
    
    chb_list = range(1,25,1)
    total_error_train = 0
    total_error_val = 0
    total_error_test = 0
    total_error_test_new = 0
    total_light_number_layers = 0
    total_percentage = [ 0 for i in range(length_network) ] #[0,0,0,0]
    print("-------------begin-------------------")
    for i in chb_list:
        print("process for patient "+str(i))
        if i== 6 or i==14 or i==16: # we do not consider patient 6/14/16
            continue
        if i <10:
            which_patients = 'chb0'+str(i)
        else:
            which_patients = 'chb'+str(i)
        train_error,val_error,test_error,test_error_light,light_number_layers,percentage = each_patient_detection(which_patients,Train_flag,layers,length_network)
        total_error_train += train_error
        total_error_val += val_error
        total_error_test += test_error
        total_error_test_new += test_error_light
        total_light_number_layers += light_number_layers
        total_percentage = [total_percentage[i] + percentage[i] for i in range(len(percentage))]

    print("-------------over-------------------")
    print("the mean error for train is "+str(total_error_train/21))
    print("the mean error for val is "+str(total_error_val/21))
    print("the mean error for test is "+str(total_error_test/21))
    print("the mean error for test_light is "+str(total_error_test_new/21))
    print("the mean number of used layers is "+str(total_light_number_layers/21))
    print("the mean percentage of used layers is "+str([x/21 for x in total_percentage]))

train_seizure(Train_flag,layers,length_network)
    

