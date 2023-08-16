import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam

import numpy as np
from sklearn.utils import shuffle

from itertools import islice

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
            self.layers += [Layer(dims[d], dims[d + 1]).cuda()]
        for d in range(1, len(dims)):
            in_dim = dims[d]
            for i in range(1, d):
                in_dim += dims[i]
            self.softmax_layers += [SoftmaxLayer(in_features=in_dim, out_features=10).cuda()]

    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def light_predict_one_sample(self, x):
        goodness_per_label = np.zeros(10)
        h_list = []
        for label in range(10):
            h_list.append(overlay_y_on_x(x, label))

        for i, layer in enumerate(self.layers, start=1):

            # L1
            if i == 1:
                for label in range(10):
                    h_list[label] = layer(h_list[label])
                    goodness_per_label[label] += h_list[label].pow(2).mean(1)
                predicted_with_layers_up_to = 1
                next_layer_flag = False
                if max(goodness_per_label) < (4.342 - 0.940):
                    next_layer_flag = True
            # L2
            elif i == 2 and next_layer_flag:
                for label in range(10):
                    h_list[label] = layer(h_list[label])
                    goodness_per_label[label] += h_list[label].pow(2).mean(1)
                predicted_with_layers_up_to += 1
                next_layer_flag = False
                if max(goodness_per_label) < (8.673 - 1.649):
                    next_layer_flag = True
            # L3
            elif i == 3 and next_layer_flag:
                for label in range(10):
                    h_list[label] = layer(h_list[label])
                    goodness_per_label[label] += h_list[label].pow(2).mean(1)
                predicted_with_layers_up_to += 1
                next_layer_flag = False
                if max(goodness_per_label) < (12.502 - 2.061):
                    next_layer_flag = True
            # L4
            elif i == 4 and  next_layer_flag:
                for label in range(10):
                    h_list[label] = layer(h_list[label])
                    goodness_per_label[label] += h_list[label].pow(2).mean(1)
                predicted_with_layers_up_to += 1
        h_list.clear()
        # print(goodness_per_label, predicted_with_layers_up_to)
        # print(goodness_per_label.argmax(0))
        return goodness_per_label.argmax(0), predicted_with_layers_up_to

    def light_predict_4l(self, x):
        goodness_per_label_l1 = []
        goodness_per_label_l1_l2 = []
        goodness_per_label_l1_l2_l3 = []
        goodness_per_label_l1_l2_l3_l4 = []
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness_l1 = []
            goodness_l1_l2 = []
            goodness_l1_l2_l3 = []
            goodness_l1_l2_l3_l4 = []
            # confidence_threshold = xyz
            for i, layer in enumerate(self.layers, start=1):
                if i==1:
                    h = layer(h)
                    goodness_l1 += [h.pow(2).mean(1)]
                    goodness_l1_l2 += [h.pow(2).mean(1)]
                    goodness_l1_l2_l3 += [h.pow(2).mean(1)]
                    goodness_l1_l2_l3_l4 += [h.pow(2).mean(1)]
                elif i==2:
                    h = layer(h)
                    goodness_l1_l2 += [h.pow(2).mean(1)]
                    goodness_l1_l2_l3 += [h.pow(2).mean(1)]
                    goodness_l1_l2_l3_l4 += [h.pow(2).mean(1)]
                elif i==3:
                    h = layer(h)
                    goodness_l1_l2_l3 += [h.pow(2).mean(1)]
                    goodness_l1_l2_l3_l4 += [h.pow(2).mean(1)]
                elif i==4:
                    h = layer(h)
                    goodness_l1_l2_l3_l4 += [h.pow(2).mean(1)]

            goodness_per_label_l1 += [sum(goodness_l1).unsqueeze(1)]
            goodness_per_label_l1_l2 += [sum(goodness_l1_l2).unsqueeze(1)]
            goodness_per_label_l1_l2_l3 += [sum(goodness_l1_l2_l3).unsqueeze(1)]
            goodness_per_label_l1_l2_l3_l4 += [sum(goodness_l1_l2_l3_l4).unsqueeze(1)]
        goodness_per_label_l1 = torch.cat(goodness_per_label_l1, 1)
        goodness_per_label_l1_l2 = torch.cat(goodness_per_label_l1_l2, 1)
        goodness_per_label_l1_l2_l3 = torch.cat(goodness_per_label_l1_l2_l3, 1)
        goodness_per_label_l1_l2_l3_l4 = torch.cat(goodness_per_label_l1_l2_l3_l4, 1)
        return goodness_per_label_l1.argmax(1), goodness_per_label_l1_l2.argmax(1), \
            goodness_per_label_l1_l2_l3.argmax(1), goodness_per_label_l1_l2_l3_l4.argmax(1), \
            goodness_per_label_l1, goodness_per_label_l1_l2, \
            goodness_per_label_l1_l2_l3, goodness_per_label_l1_l2_l3_l4

    def light_predict_2l(self, x):
        goodness_per_label_l1 = []
        goodness_per_label_l1_l2 = []
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness_l1 = []
            goodness_l1_l2 = []
            # confidence_threshold = xyz
            for i, layer in enumerate(self.layers, start=1):
                if i==1:
                    h = layer(h)
                    goodness_l1 += [h.pow(2).mean(1)]
                    goodness_l1_l2 += [h.pow(2).mean(1)]
                elif i==2:
                    h = layer(h)
                    goodness_l1_l2 += [h.pow(2).mean(1)]
            goodness_per_label_l1 += [sum(goodness_l1).unsqueeze(1)]
            goodness_per_label_l1_l2 += [sum(goodness_l1_l2).unsqueeze(1)]
        goodness_per_label_l1 = torch.cat(goodness_per_label_l1, 1)
        goodness_per_label_l1_l2 = torch.cat(goodness_per_label_l1_l2, 1)
        return goodness_per_label_l1.argmax(1), goodness_per_label_l1_l2.argmax(1)

    def train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            # print('training layer', i, '...')
            h_pos, h_neg = layer.train(h_pos, h_neg)

    def train_softmax_layer(self, x_neutral_label, y, batch_size, dims):  # , num_layers, num_neurons
        h_neutral_label = x_neutral_label

        for d, softmax_layer in enumerate(self.softmax_layers, start=0):
            # for softmax layer of layer d
            num_input_features = sum(dims[1:(d + 2)])
            softmax_layer_input = torch.empty((batch_size, num_input_features))
            for i, layer in islice(enumerate(self.layers), 0, (d + 1)):  # from first layer to layer d (d included)
                h_neutral_label = layer.forward(h_neutral_label)
                # store the result in softmax_layer_input
                index_start = sum(dims[1:(i + 1)])
                index_end = index_start + dims[i + 1]
                softmax_layer_input[:, index_start:index_end] = h_neutral_label
            self.softmax_layer.train(softmax_layer_input, y)


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
        return self.relu(
            torch.mm(x_direction, self.weight.T) +
            self.bias.unsqueeze(0))

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
        temp = self.softmax_l(x)
        output = self.softmax(temp)  # .argmax(1)
        return output

    def to_categorical(y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]

    def train(self, x, y):
        self.opt.zero_grad()
        yhat = self.forward(x)
        # y_one_hot = nn.functional.one_hot(y, num_classes=10).to(torch.float32)
        loss = self.criterion(yhat.cuda(), y)
        loss.backward()
        self.opt.step()


def build_model(x_pos, x_neg, x_neutral, targets):
    # torch.manual_seed(1234)
    dims = [784, 500, 500]  # 2000, 2000, 2000, 2000
    model = Net(dims)
    # model = Net([784, 2000, 2000, 2000, 2000])

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

    # training the softmax layer
    for epoch in tqdm(range(num_epochs)):

        train_data_record_indices = range(0, 60000)
        train_data_record_indices_shuffled = shuffle(train_data_record_indices)

        batch_size = 500
        num_batches = int(60000 / batch_size)
        chunk_indices = np.array_split(train_data_record_indices_shuffled, num_batches)
        for i in range(num_batches):
            x_neutral_, targets_ = x_neutral[chunk_indices[i]], targets[chunk_indices[i]]
            model.train_softmax_layer(x_neutral_, targets_, batch_size, dims)  # , num_layers, num_neurons

    # save model
    name = 'temp'  # '4L_2kN_100E_500B' '2L_500N_100E_5kB_50kS' '2L_500N_10E_500B_50kS'
    torch.save(model, 'model/' + name)
