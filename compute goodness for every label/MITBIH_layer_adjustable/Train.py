import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam

import numpy as np
from sklearn.utils import shuffle

import os

def overlay_y_on_x(x, y):
    """Replace the first 5 pixels of data [x] with one-hot-encoded label [y]
    """
    x_ = x.clone()
    x_[:, :5] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_


class Net(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = []
        for d in range(len(dims) - 1):
            #self.layers += [Layer(dims[d], dims[d + 1]).cuda()]
            self.layers += [Layer(dims[d], dims[d + 1])]

    def predict(self, x):
        goodness_per_label = []
        for label in range(5):
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def light_predict_one_sample(self, x,means,stds,length_network):
        goodness_per_label = np.zeros(5)
        h_list = []
        for label in range(5):
            h_list.append(overlay_y_on_x(x, label))

        next_layer_flag = True
        predicted_with_layers_up_to = 0
        for i, layer in enumerate(self.layers, start=1):

            for j in range(length_network):
                if i == j+1 and next_layer_flag:
                    for label in range(5):
                        h_list[label] = layer(h_list[label])
                        goodness_per_label[label] += h_list[label].pow(2).mean(1)
                    predicted_with_layers_up_to += 1
                    next_layer_flag = False
                    if max(goodness_per_label) < (means[i-1] - stds[i-1]):
                        next_layer_flag = True
        h_list.clear()
        # print(goodness_per_label, predicted_with_layers_up_to)
        # print(goodness_per_label.argmax(0))
        return goodness_per_label.argmax(0), predicted_with_layers_up_to
    
    def light_predict_4l(self, x,length_network):
        goodness_per_label = [ [ ] for i in range(length_network) ]

        for label in range(5):
            h = overlay_y_on_x(x, label)
            goodness = [ [ ] for i in range(length_network) ]

            for i, layer in enumerate(self.layers, start=1):
                h = layer(h)
                for j in range(i,length_network+1):
                    goodness[j-1] +=[ h.pow(2).mean(1)]
            
            for i in range(length_network):
                goodness_per_label[i] += [sum(goodness[i]).unsqueeze(1)]  

        goodness_final_label  = [ [ ] for i in range(length_network) ] 
        for i in range(length_network):
            goodness_per_label[i] = torch.cat(goodness_per_label[i], 1)
            goodness_final_label[i] = goodness_per_label[i].argmax(1).unsqueeze(0)

        #temp1 = torch.cat(goodness_final_label,0)
        return torch.cat(goodness_final_label,0),goodness_per_label
    
    def train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            # print('training layer', i, '...')
            h_pos, h_neg = layer.train(h_pos, h_neg)


class Layer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=0.03)
        self.threshold = 2.0
        self.num_epochs = 1

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(
            torch.mm(x_direction, self.weight.T.to(torch.float64)) +
            self.bias.to(torch.float64).unsqueeze(0))

    def train(self, x_pos, x_neg):
        for i in range(self.num_epochs):
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




def build_model(x_pos, x_neg,layers):
    # torch.manual_seed(1234)
    model = Net(layers) 

    Num_Epochs = 100
    for epoch in tqdm(range(Num_Epochs)):

        train_data_record_indices = range(0, 87554-10000)
        train_data_record_indices_shuffled = shuffle(train_data_record_indices)

        batch_size = 500
        num_batches = int((87554-10000) / batch_size)
        chunk_indices = np.array_split(train_data_record_indices_shuffled, num_batches)
        for i in range(num_batches):
            x_pos_, x_neg_ = x_pos[chunk_indices[i]], x_neg[chunk_indices[i]]
            model.train(x_pos_, x_neg_)

    # save model
    #name = 'temp' #'4L_2kN_100E_500B'  
    #torch.save(model, os.path.split(os.path.realpath(__file__))[0]+'/model/' + name)
    return model