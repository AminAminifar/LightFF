import torch

from sklearn.metrics import f1_score, accuracy_score
import numpy as np

import tools

def print_results(labels_vec, predictions_vec):
    f1_performance = f1_score(labels_vec, predictions_vec, average='macro')
    acc_performance = accuracy_score(labels_vec, predictions_vec)

    print("\tF1-score: ", f1_performance)
    print("\tAccuracy: ", acc_performance)


def eval_train_set(model, inputs, targets):
    # train set
    num_train_samples = 50000
    train_data_record_indices = range(0, num_train_samples)

    batch_size = 5000
    num_batches = int(num_train_samples / batch_size)
    chunk_indices = np.array_split(train_data_record_indices, num_batches)

    y_predicted = np.zeros(num_train_samples)
    for i in range(num_batches):
        x_pos_ = inputs[chunk_indices[i]]
        y_predicted[chunk_indices[i]] = model.predict(x_pos_).detach().cpu().numpy()

    print("\nResults for the {}TRAIN{} set: ".format('\033[1m', '\033[0m'))
    print_results(targets.detach().cpu().numpy(), y_predicted)
    print('\tError:', 1.0 - torch.eq(torch.tensor(y_predicted), targets.detach().cpu()).float().mean().item())


def eval_test_set(model, inputs, targets):
    # test set
    num_test_samples = 10000
    test_data_record_indices = range(0, num_test_samples)

    batch_size = 5000
    num_batches = int(num_test_samples / batch_size)
    chunk_indices_test = np.array_split(test_data_record_indices, num_batches)
    y_predicted = np.zeros(num_test_samples)
    for i in range(num_batches):
        x_pos_ = inputs[chunk_indices_test[i]]
        y_predicted[chunk_indices_test[i]] = model.predict(x_pos_).detach().cpu().numpy()

    print("\nResults for the {}TEST{} set: ".format('\033[1m', '\033[0m'))
    print_results(targets.detach().cpu().numpy(), y_predicted)
    print('\tError:', 1.0 - torch.eq(torch.tensor(y_predicted), targets.detach().cpu()).float().mean().item())


def eval_val_set(model, inputs, targets):
    # test set
    num_test_samples = 10000
    test_data_record_indices = range(0, num_test_samples)

    batch_size = 5000
    num_batches = int(num_test_samples / batch_size)
    chunk_indices_validation = np.array_split(test_data_record_indices, num_batches)
    y_predicted = np.zeros(num_test_samples)
    for i in range(num_batches):
        x_pos_ = inputs[chunk_indices_validation[i]]
        y_predicted[chunk_indices_validation[i]] = model.predict(x_pos_).detach().cpu().numpy()

    print("\nResults for the {}VALIDATION{} set: ".format('\033[1m', '\033[0m'))
    print_results(targets.detach().cpu().numpy(), y_predicted)
    print('\tError:', 1.0 - torch.eq(torch.tensor(y_predicted), targets.detach().cpu()).float().mean().item())


def analysis_val_set(model, inputs, targets):
    # test set
    num_test_samples = 10000
    test_data_record_indices = range(0, num_test_samples)

    batch_size = 1000
    num_batches = int(num_test_samples / batch_size)
    chunk_indices_validation = np.array_split(test_data_record_indices, num_batches)
    y_predicted_l1 = np.zeros(num_test_samples)
    y_predicted_l1_l2 = np.zeros(num_test_samples)
    y_predicted_l1_l2_l3 = np.zeros(num_test_samples)
    y_predicted_l1_l2_l3_l4 = np.zeros(num_test_samples)
    goodness_per_label_l1, goodness_per_label_l1_l2, goodness_per_label_l1_l2_l3, goodness_per_label_l1_l2_l3_l4 = \
        np.zeros((num_test_samples, 10)), np.zeros((num_test_samples, 10)), \
        np.zeros((num_test_samples, 10)), np.zeros((num_test_samples, 10))
    for i in range(num_batches):
        x_pos_ = inputs[chunk_indices_validation[i]]

        temp_l1, temp_l1_l2, temp_l1_l2_l3, temp_l1_l2_l3_l4, \
        temp_goodness_per_label_l1, temp_goodness_per_label_l1_l2, \
        temp_goodness_per_label_l1_l2_l3, temp_goodness_per_label_l1_l2_l3_l4 = model.light_predict(x_pos_)

        y_predicted_l1[chunk_indices_validation[i]], y_predicted_l1_l2[chunk_indices_validation[i]], \
            y_predicted_l1_l2_l3[chunk_indices_validation[i]], y_predicted_l1_l2_l3_l4[chunk_indices_validation[i]] = \
            temp_l1.detach().cpu().numpy(), temp_l1_l2.detach().cpu().numpy(), \
            temp_l1_l2_l3.detach().cpu().numpy(), temp_l1_l2_l3_l4.detach().cpu().numpy()

        print("__>", np.shape(temp_goodness_per_label_l1.detach().cpu().numpy()), np.shape(goodness_per_label_l1[chunk_indices_validation[i],:]))
        # exit()

        goodness_per_label_l1[chunk_indices_validation[i], :], goodness_per_label_l1_l2[chunk_indices_validation[i], :], \
        goodness_per_label_l1_l2_l3[chunk_indices_validation[i], :], goodness_per_label_l1_l2_l3_l4[chunk_indices_validation[i], :] = \
            temp_goodness_per_label_l1.detach().cpu().numpy(), temp_goodness_per_label_l1_l2.detach().cpu().numpy(), \
            temp_goodness_per_label_l1_l2_l3.detach().cpu().numpy(), temp_goodness_per_label_l1_l2_l3_l4.detach().cpu().numpy()

    print("\nResults for the {}VALIDATION{} set for {}Layer 1{}: ".
          format('\033[1m', '\033[0m', '\033[1m', '\033[0m'))
    print_results(targets.detach().cpu().numpy(), y_predicted_l1)
    print('\tError:', 1.0 - torch.eq(torch.tensor(y_predicted_l1), targets.detach().cpu()).float().mean().item())

    print("\nResults for the {}VALIDATION{} set for {}Layers 1 and 2{}: ".
          format('\033[1m', '\033[0m', '\033[1m', '\033[0m'))
    print_results(targets.detach().cpu().numpy(), y_predicted_l1_l2)
    print('\tError:', 1.0 - torch.eq(torch.tensor(y_predicted_l1_l2), targets.detach().cpu()).float().mean().item())

    print("\nResults for the {}VALIDATION{} set for {}Layers 1 and 2 and 3{}: ".
          format('\033[1m', '\033[0m', '\033[1m', '\033[0m'))
    print_results(targets.detach().cpu().numpy(), y_predicted_l1_l2_l3)
    print('\tError:', 1.0 - torch.eq(torch.tensor(y_predicted_l1_l2_l3), targets.detach().cpu()).float().mean().item())

    print("\nResults for the {}VALIDATION{} set for {}Layers 1 and 2 and 3 and 4{}: ".
          format('\033[1m', '\033[0m', '\033[1m', '\033[0m'))
    print_results(targets.detach().cpu().numpy(), y_predicted_l1_l2_l3_l4)
    print('\tError:', 1.0 - torch.eq(torch.tensor(y_predicted_l1_l2_l3_l4), targets.detach().cpu()).float().mean().item())

    # tools.plot_goodness_distributions(goodness_per_label_l1, goodness_per_label_l1_l2,
    #                                   goodness_per_label_l1_l2_l3, goodness_per_label_l1_l2_l3_l4)

    # print(np.shape(goodness_per_label_l1[:100, :]))
    t = targets.detach().cpu().numpy()
    # print(np.shape(t[:100]))
    tools.plot_goodness_distributions(goodness_per_label_l1, t)  # goodness_per_label_l1[:1000, :], t[:1000]


def analysis_val_set_2l(model, inputs, targets):
    # test set
    num_test_samples = 10000
    test_data_record_indices = range(0, num_test_samples)

    batch_size = 5000
    num_batches = int(num_test_samples / batch_size)
    chunk_indices_validation = np.array_split(test_data_record_indices, num_batches)
    y_predicted_l1 = np.zeros(num_test_samples)
    y_predicted_l1_l2 = np.zeros(num_test_samples)
    for i in range(num_batches):
        x_pos_ = inputs[chunk_indices_validation[i]]
        temp_l1, temp_l1_l2 = model.light_predict_2l(x_pos_)
        y_predicted_l1[chunk_indices_validation[i]], y_predicted_l1_l2[chunk_indices_validation[i]] =\
            temp_l1.detach().cpu().numpy(), temp_l1_l2.detach().cpu().numpy()

    print("\nResults for the {}VALIDATION{} set for {}Layer 1{}: ".
          format('\033[1m', '\033[0m', '\033[1m', '\033[0m'))
    print_results(targets.detach().cpu().numpy(), y_predicted_l1)
    print('\tError:', 1.0 - torch.eq(torch.tensor(y_predicted_l1), targets.detach().cpu()).float().mean().item())

    print("\nResults for the {}VALIDATION{} set for {}Layers 1 and 2{}: ".
          format('\033[1m', '\033[0m', '\033[1m', '\033[0m'))
    print_results(targets.detach().cpu().numpy(), y_predicted_l1_l2)
    print('\tError:', 1.0 - torch.eq(torch.tensor(y_predicted_l1_l2), targets.detach().cpu()).float().mean().item())


