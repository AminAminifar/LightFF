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
def MITBIH_loaders():
    import mitbih_dataset.load_data as bih
    balance = 1 # 0 means no balanced (raw data), and 1 means balanced (weighted selected).
    # Please see .mithib_dataset/Distribution.png for more data structure and distribution information.
    # The above .png is from the paper-Zhang, Dengqing, et al. "An ECG heartbeat classification method based on deep convolutional neural network." Journal of Healthcare Engineering 2021 (2021): 1-9.
    x_train, y_train, x_test, y_test = bih.load_data(balance)

    x_train = x_train[:,:169,:].reshape((x_train.shape[0],13,13))
    x_test = x_test[:,:169,:].reshape((x_test.shape[0],13,13))
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train=x_train.reshape(x_train.shape[0], -1)
    x_test=x_test.reshape(x_test.shape[0], -1)

    train_data=torch.utils.data.TensorDataset(torch.tensor(x_train).float(),torch.tensor(y_train).long())
    test_data=torch.utils.data.TensorDataset(torch.tensor(x_test).float(),torch.tensor(y_test).long())



    train_loader = torch.utils.data.DataLoader(train_data,batch_size=87554,shuffle=True,pin_memory=True,num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=21892,shuffle=True,pin_memory=True,num_workers=0)

    return train_loader, test_loader



train_loader, test_loader = MITBIH_loaders()

# train data
inputs, targets = next(iter(train_loader))
inputs, targets = inputs.to(device), targets.to(device)

# create a validation set
X_train, X_val, y_train, y_val = train_test_split(inputs, targets, test_size=10000, random_state=0)

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

