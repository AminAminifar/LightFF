import torch

from sklearn.metrics import f1_score, accuracy_score
import numpy as np


def print_results(labels_vec, predictions_vec):
    f1_performance = f1_score(labels_vec, predictions_vec, average='macro')
    acc_performance = accuracy_score(labels_vec, predictions_vec)

    print("f1_performance: ", f1_performance)
    print("acc_performance", acc_performance)


def eval_train_set(model, inputs, targets):
    # train set
    train_data_record_indices = range(0, 60000)

    batch_size = 500
    num_batches = int(60000 / batch_size)
    chunk_indices = np.array_split(train_data_record_indices, num_batches)

    y_predicted = np.zeros((60000))
    for i in range(num_batches):
        x_pos_ = inputs[chunk_indices[i]]
        y_predicted[chunk_indices[i]] = model.predict(x_pos_).detach().cpu().numpy()

    print("\n Results for the train set: ")
    print_results(targets.detach().cpu().numpy(), y_predicted)
    print('train error:', 1.0 - torch.eq(torch.tensor(y_predicted), targets.detach().cpu()).float().mean().item())


def eval_test_set(model, inputs, targets):
    # test set
    test_data_record_indices = range(0, 10000)

    batch_size = 500
    num_batches = int(10000 / batch_size)
    chunk_indices_test = np.array_split(test_data_record_indices, num_batches)
    y_predicted = np.zeros((10000))
    for i in range(num_batches):
        x_pos_ = inputs[chunk_indices_test[i]]
        y_predicted[chunk_indices_test[i]] = model.predict(x_pos_).detach().cpu().numpy()

    print("\n Results for the test set: ")
    print_results(targets.detach().cpu().numpy(), y_predicted)
    print('test error:', 1.0 - torch.eq(torch.tensor(y_predicted), targets.detach().cpu()).float().mean().item())

