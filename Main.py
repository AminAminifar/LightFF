import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
import numpy as np
from sklearn.utils import shuffle

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
            torch.mm(x_direction, self.weight.T) +
            self.bias.unsqueeze(0))

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

    
def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize = (4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()
    
    
if __name__ == "__main__":
    torch.manual_seed(1234)
    model = Net([784, 500, 500])

    train_loader, test_loader = MNIST_loaders()

    inputs, targets = next(iter(train_loader))
    inputs, targets = inputs.cuda(), targets.cuda()
    x_pos = overlay_y_on_x(inputs, targets)
    rnd = torch.randperm(inputs.size(0))
    x_neg = overlay_y_on_x(inputs, targets[rnd])

    for epoch in tqdm(range(100)):

        train_data_record_indices = range(0, 60000)
        train_data_record_indices_shuffled = shuffle(train_data_record_indices)

        batch_size = 5000
        num_batches = int(60000 / batch_size)
        chunk_indices = np.array_split(train_data_record_indices_shuffled, num_batches)
        for i in range(num_batches):
            x_pos_, x_neg_ = x_pos[chunk_indices[i]], x_neg[chunk_indices[i]]
            model.train(x_pos_, x_neg_)


    def print_results(labels_vec, predictions_vec):
        # tn, fp, fn, tp = confusion_matrix(labels_vec, predictions_vec).ravel()
        f1_performance = f1_score(labels_vec, predictions_vec, average='macro')
        acc_performance = accuracy_score(labels_vec, predictions_vec)
        mcc_performance = matthews_corrcoef(labels_vec, predictions_vec)
        # print("tn, fp, fn, tp: ", tn, fp, fn, tp)
        print("f1_performance: ", f1_performance)
        print("acc_performance", acc_performance)
        print("mcc_performance: ", mcc_performance)


    print("Results for the test set: ")
    y_predicted = np.zeros((60000))
    for i in range(num_batches):
        x_pos_ = x_pos[chunk_indices[i]]
        y_predicted[chunk_indices[i]] = model.predict(x_pos_).detach().cpu().numpy()
    print_results(targets.detach().cpu().numpy(), y_predicted)
    print('train error:', 1.0 - torch.eq(torch.tensor(y_predicted), targets.detach().cpu()).float().mean().item())
    # print(torch.eq(torch.tensor(y_predicted), targets.detach().cpu()))

    x_te, y_te = next(iter(test_loader))
    x_te, y_te = x_te.cuda(), y_te.cuda()

    print("Results for the test set: ")
    y_predicted = model.predict(x_te).detach().cpu().numpy()
    print_results(y_te.detach().cpu().numpy(), y_predicted)
    print('test error:', 1.0 - model.predict(x_te).eq(y_te).float().mean().item())
    # print(model.predict(x_te).eq(y_te).float())