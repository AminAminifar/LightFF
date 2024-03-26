import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam

import numpy as np
from sklearn.utils import shuffle

from itertools import islice

import TrainBPmodel



if torch.cuda.is_available():
    device = torch.device("cuda")

else:
    device = torch.device("cpu")


class Net(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = []
        self.softmax_layers = []
        for d in range(len(dims) - 1):
            self.layers += [Layer(dims[d], dims[d + 1]).to(device)]
        for d in range(1, len(dims)):
            in_dim = dims[d]
            for i in range(1, d):
                in_dim += dims[i]
            self.softmax_layers += [SoftmaxLayer(in_features=in_dim, out_features=10)]  # .cuda()

    def predict_one_pass(self, x, batch_size):
        num_layers = len(self.layers)
        # val set
        h = x

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
        h = x

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
        h = x
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

            softmax_layer_output_l, softmax_layer_output = softmax_layer(softmax_layer_input)
            y_predicted_on_layer[i, :] = softmax_layer_output.argmax(1)  # to be checked
            softmax_output_on_layer[i, :, :] = softmax_layer_output_l.detach().cpu().numpy()
        # print(y_predicted_on_layer.shape)
        # exit()

        return y_predicted_on_layer, cumulative_goodness_on_layer, softmax_output_on_layer

    def train(self, BPmodel):
        # keys = list(BPmodel.state_dict().keys())
        for i, layer in enumerate(self.layers):
        #     with torch.no_grad(): # does not properly work yet
        #         layer.weight.copy_(BPmodel.state_dict()[keys[2*i]])
        #         layer.bias.copy_(BPmodel.state_dict()[keys[2*i + 1]])

            # quick and dirty
            if i == 0:
                with torch.no_grad():
                    layer.weight.copy_(BPmodel.fc1[0].weight)
                    layer.bias.copy_(BPmodel.fc1[0].bias)
            elif i == 1:
                with torch.no_grad():
                    layer.weight.copy_(BPmodel.fc2[0].weight)
                    layer.bias.copy_(BPmodel.fc2[0].bias)
            elif i == 2:
                with torch.no_grad():
                    layer.weight.copy_(BPmodel.fc3[0].weight)
                    layer.bias.copy_(BPmodel.fc3[0].bias)
            elif i == 3:
                with torch.no_grad():
                    layer.weight.copy_(BPmodel.fc4[0].weight)
                    layer.bias.copy_(BPmodel.fc4[0].bias)
            elif i == 4:
                with torch.no_grad():
                    layer.weight.copy_(BPmodel.fc6[0].weight)
                    layer.bias.copy_(BPmodel.fc6[0].bias)
            elif i == 5:
                with torch.no_grad():
                    layer.weight.copy_(BPmodel.fc7[0].weight)
                    layer.bias.copy_(BPmodel.fc7[0].bias)

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
    def my_block(self, in_size, out_size):
        block = torch.nn.Sequential(
            nn.Linear(in_size, out_size),
            # torch.nn.BatchNorm1d(out_size),
            torch.nn.ReLU(),
        )
        return block

    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features)
        self.fc = nn.Linear(in_features, out_features)
        # self.nr = torch.nn.BatchNorm1d(out_features)
        self.relu = torch.nn.ReLU()

    def forward(self, x):  # should we change it?
        x = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(self.fc(x))

    # def forward(self, x):
    #     x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
    #     return self.relu(torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0))

class SoftmaxLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.softmax_l = nn.Linear(in_features, out_features)
        # nn.init.xavier_uniform_(self.softmax_l.weight)
        self.softmax = torch.nn.Softmax(dim=1)
        self.opt = Adam(self.parameters(), lr=0.03)  # 0.03  0.0001
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        #  x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        output_l = self.softmax_l(x)
        output = self.softmax(output_l)  # .argmax(1)
        return output_l, output

    # def forward_l(self, x):
    #     #  x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
    #     output = self.softmax_l(x)
    #     return output

    def to_categorical(y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]

    def train(self, x, y):
        self.opt.zero_grad()
        _, yhat = self.forward(x)
        # y_one_hot = nn.functional.one_hot(y, num_classes=10).to(torch.float32)
        loss = self.criterion(yhat.to(device), y)
        loss.backward()
        self.opt.step()


def build_model(x, targets):
    # torch.manual_seed(1234)
    dims = [169, 2000, 2000, 2000, 2000,2000,2000]  # 2000, 2000, 2000, 2000
    model = Net(dims)
    # model = Net([784, 2000, 2000, 2000, 2000])

    BPmodel = TrainBPmodel.train_BPmodel(X=x,Y=targets)
    model.train(BPmodel)

    num_epochs = 1
    # training the softmax layer
    for epoch in tqdm(range(num_epochs)):

        train_data_record_indices = range(0, 87500)
        train_data_record_indices_shuffled = shuffle(train_data_record_indices)

        batch_size = 500
        num_batches = int(87500 / batch_size)
        chunk_indices = np.array_split(train_data_record_indices_shuffled, num_batches)
        for i in range(num_batches):
            x_neutral_, targets_ = x[chunk_indices[i]], targets[chunk_indices[i]]
            model.train_softmax_layer(x_neutral_, targets_, batch_size, dims)  # , num_layers, num_neurons

    # save model
    name = 'LFFmodel'  # '4L_2kN_100E_500B' '2L_500N_100E_5kB_50kS' '2L_500N_10E_500B_50kS'
    torch.save(model, 'model/' + name)

