import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam

import numpy as np
from sklearn.utils import shuffle

from itertools import islice
import os

def overlay_y_on_x(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_


def overlay_on_x_neutral(x):
    """Replace the first 10 pixels of data [x] with 0.1s
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), :10] = 0.1  # x.max()
    return x_


class Net(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = []
        self.softmax_layers = []
        for d in range(len(dims) - 1):
            self.layers += [Layer(dims[d], dims[d + 1])]
        for d in range(1, len(dims)):
            in_dim = dims[d]
            for i in range(1, d):
                in_dim += dims[i]
            self.softmax_layers += [SoftmaxLayer(in_features=in_dim, out_features=10)]  # .cuda()



    def predict_one_pass(self, x, batch_size):
        num_layers = len(self.layers)
        # val set
        h = overlay_on_x_neutral(x)

        for i, (layer, softmax_layer) in enumerate(zip(self.layers, self.softmax_layers), start=0):
            h = layer(h)

            try:
                softmax_layer_input
                softmax_layer_input = torch.cat((softmax_layer_input, h.cpu()), 1)
                # print("in try: ", softmax_layer_input.size(), "i: ", i)  # temp
            except NameError:
                softmax_layer_input = h.cpu()
                # print("in except: ", softmax_layer_input.size(), "i: ", i)  # temp
            if i == num_layers - 1:
                _, softmax_layer_output = softmax_layer(softmax_layer_input)

        output = softmax_layer_output.argmax(1)
        # print("output: ", output)
        return output

    def check_confidence(self, layer_num, confidence_mean_vec, confidence_std_vec, softmax_layer_output_l):
        confidence_flag = False
        threshold = confidence_mean_vec[layer_num] - confidence_std_vec[layer_num]
        if torch.max(softmax_layer_output_l) > threshold:  # then we are confident
            confidence_flag = True
        return confidence_flag

    def light_predict_one_sample(self, x, confidence_mean_vec, confidence_std_vec):
        h = overlay_on_x_neutral(x)

        confidence_flag = False  # if confident: True
        predicted_with_layers_up_to = 0
        for i, (layer, softmax_layer) in enumerate(zip(self.layers, self.softmax_layers), start=0):
            if not confidence_flag:
                predicted_with_layers_up_to += 1
                h = layer(h)

                try:
                    softmax_layer_input
                    softmax_layer_input = torch.cat((softmax_layer_input, h.cpu()), 1)
                    # print("in try: ", softmax_layer_input.size(), "i: ", i)  # temp
                except NameError:
                    softmax_layer_input = h.cpu()
                    # print("in except: ", softmax_layer_input.size(), "i: ", i)  # temp

                softmax_layer_output_l, softmax_layer_output = softmax_layer(softmax_layer_input)

                # check confidence
                # not required for the last layer
                confidence_flag = self.check_confidence(layer_num=i, confidence_mean_vec=confidence_mean_vec,
                                                        confidence_std_vec=confidence_std_vec,
                                                        softmax_layer_output_l=softmax_layer_output_l)

        return softmax_layer_output.argmax(1), predicted_with_layers_up_to

    def light_predict_analysis(self, x, num_layers):  # dims is not needed; just num layers
        num_samples = x.shape[0]

        y_predicted_on_layer = np.zeros((num_layers, num_samples))
        cumulative_goodness_on_layer = np.zeros((num_layers, num_samples))
        softmax_output_on_layer = np.zeros((num_layers, num_samples, 10))  # 10 is the number of softmax neurons

        # embed neutral label
        h = overlay_on_x_neutral(x)
        # softmax_input_size = 0
        for i, (layer, softmax_layer) in enumerate(zip(self.layers, self.softmax_layers), start=0):
            h = layer(h)  # should be the same as forward
            # softmax_input_size += h.size()[1]
            # softmax_layer_input = torch.empty((num_samples, softmax_input_size))
            try:
                softmax_layer_input
                softmax_layer_input = torch.cat((softmax_layer_input, h.cpu()), 1)
                # print("in try: ", softmax_layer_input.size(), "i: ", i)  # temp
            except NameError:
                softmax_layer_input = h.cpu()
                # print("in except: ", softmax_layer_input.size(), "i: ", i)  # temp

            for j in range(i, num_layers):
                cumulative_goodness_on_layer[j, :] += h.pow(2).mean(1).detach().cpu().numpy()

            # print(softmax_layer(softmax_layer_input).shape)
            # print(softmax_layer(softmax_layer_input).detach().cpu().numpy())
            # y_predicted_on_layer[i, :] = softmax_layer(softmax_layer_input).argmax(1)  # to be checked
            softmax_layer_output_l, softmax_layer_output = softmax_layer(softmax_layer_input)
            y_predicted_on_layer[i, :] = softmax_layer_output.argmax(1)  # to be checked
            # softmax_output_on_layer[i, :, :] = softmax_layer(softmax_layer_input).detach().cpu().numpy()
            # softmax_output_on_layer[i, :, :] = softmax_layer.forward(softmax_layer_input).detach().cpu().numpy()
            softmax_output_on_layer[i, :, :] = softmax_layer_output_l.detach().cpu().numpy()
        # print(y_predicted_on_layer.shape)
        # exit()

        return y_predicted_on_layer, cumulative_goodness_on_layer, softmax_output_on_layer

    def train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            # print('training layer', i, '...')
            h_pos, h_neg = layer.train(h_pos, h_neg)

    def train_softmax_layer(self, x_neutral_label, y, batch_size, dims):  # , num_layers, num_neurons

        for d, softmax_layer in enumerate(self.softmax_layers, start=0):
            h_neutral_label = x_neutral_label
            # for softmax layer of layer d
            num_input_features = sum(dims[1:(d + 2)])
            softmax_layer_input = torch.empty((batch_size, num_input_features))
            for i, layer in islice(enumerate(self.layers), 0, (d + 1)):  # from first layer to layer d (d included)
                # print("i was here ", i, d)
                h_neutral_label = layer.forward(h_neutral_label)
                # store the result in softmax_layer_input
                index_start = sum(dims[1:(i + 1)])
                index_end = index_start + dims[i + 1]
                softmax_layer_input[:, index_start:index_end] = h_neutral_label
            softmax_layer.train(softmax_layer_input, y)


class Layer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=0.03)
        self.threshold = 2.0
        self.num_iterations = 1

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0))

    def train(self, x_pos, x_neg):
        for i in range(self.num_iterations):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            # The following loss pushes pos (neg) samples to
            # values larger (smaller) than the self.threshold.
            loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold]))).mean()
            self.opt.zero_grad()
            # this backward just compute the derivative and hence
            # is not considered backpropagation.
            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()


class SoftmaxLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.softmax_l = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.softmax_l.weight)
        self.softmax = torch.nn.Softmax(dim=1)
        self.opt = Adam(self.parameters(), lr=0.03)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        #  x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        output_l = self.softmax_l(x)
        output = self.softmax(output_l)  # .argmax(1)
        return output_l, output

    def to_categorical(y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]

    def train(self, x, y):
        self.opt.zero_grad()
        yhat,_ = self.forward(x)
        # y_one_hot = nn.functional.one_hot(y, num_classes=10).to(torch.float32)
        loss = self.criterion(yhat, y)
        loss.backward()
        self.opt.step()


def build_model(x_pos, x_neg, x_neutral, targets,layers):
    # torch.manual_seed(1234)
    dims = layers  
    model = Net(dims)

    num_epochs = 100
    for epoch in tqdm(range(num_epochs)):
        num_train_samples = 50000  # 60000
        train_data_record_indices = range(0, num_train_samples)
        train_data_record_indices_shuffled = shuffle(train_data_record_indices)

        batch_size = 5000
        num_batches = int(num_train_samples / batch_size)
        chunk_indices = np.array_split(train_data_record_indices_shuffled, num_batches)
        for i in range(num_batches):
            x_pos_, x_neg_ = x_pos[chunk_indices[i]], x_neg[chunk_indices[i]]
            model.train(x_pos_, x_neg_)

    # num_epochs = 100
    # training the softmax layer
    for epoch in tqdm(range(num_epochs)):

        train_data_record_indices = range(0, 50000)
        train_data_record_indices_shuffled = shuffle(train_data_record_indices)

        batch_size = 500
        num_batches = int(50000 / batch_size)
        chunk_indices = np.array_split(train_data_record_indices_shuffled, num_batches)
        for i in range(num_batches):
            x_neutral_, targets_ = x_neutral[chunk_indices[i]], targets[chunk_indices[i]]
            model.train_softmax_layer(x_neutral_, targets_, batch_size, dims)  # , num_layers, num_neurons

    # save model
    name = 'temp_'  
    torch.save(model, os.path.split(os.path.realpath(__file__))[0]+'/model/' + name)
    return model
