import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam

import numpy as np
from sklearn.utils import shuffle


def overlay_y_on_x(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    x_ = x.clone()
    ch1_i = 0
    ch2_i = 1024
    ch3_i = 2048
    x_[:, ch1_i:ch1_i + 10] *= 0.0
    x_[:, ch2_i:ch2_i + 10] *= 0.0
    x_[:, ch3_i:ch3_i + 10] *= 0.0
    x_[range(x.shape[0]), ch1_i + y] = x.max()
    x_[range(x.shape[0]), ch2_i + y] = x.max()
    x_[range(x.shape[0]), ch3_i + y] = x.max()
    return x_


class Net(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [Layer(dims[d], dims[d + 1]).cuda()]

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
                if max(goodness_per_label) < (2.540 - 0.869):
                    next_layer_flag = True
            # L2
            elif i == 2 and next_layer_flag:
                for label in range(10):
                    h_list[label] = layer(h_list[label])
                    goodness_per_label[label] += h_list[label].pow(2).mean(1)
                predicted_with_layers_up_to += 1
                next_layer_flag = False
                if max(goodness_per_label) < (5.166 - 1.657):
                    next_layer_flag = True
            # L3
            elif i == 3 and next_layer_flag:
                for label in range(10):
                    h_list[label] = layer(h_list[label])
                    goodness_per_label[label] += h_list[label].pow(2).mean(1)
                predicted_with_layers_up_to += 1
                next_layer_flag = False
                if max(goodness_per_label) < (7.577 - 2.086):
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


def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize=(4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()


def build_model(x_pos, x_neg):
    # torch.manual_seed(1234)
    model = Net([3072, 500, 500, 500, 500])  # 2000, 2000, 2000, 2000
    # model = Net([784, 2000, 2000, 2000, 2000])

    num_epochs = 100
    for epoch in tqdm(range(num_epochs)):
        num_train_samples = 45000  # 50000
        train_data_record_indices = range(0, num_train_samples)
        train_data_record_indices_shuffled = shuffle(train_data_record_indices)

        batch_size = 5000
        num_batches = int(num_train_samples / batch_size)
        chunk_indices = np.array_split(train_data_record_indices_shuffled, num_batches)
        for i in range(num_batches):
            x_pos_, x_neg_ = x_pos[chunk_indices[i]], x_neg[chunk_indices[i]]
            model.train(x_pos_, x_neg_)

    # save model
    name = 'temp'  # '4L_2kN_100E_500B' '2L_500N_100E_5kB_50kS' '2L_500N_10E_500B_50kS'
    torch.save(model, 'model/' + name)
