import numpy as np

import torch
from Evaluation import print_results


def calculate_goodness_distributions(matrix, y_predicted_on_layer, targets, num_layers):  # matrix is softmax_output_on_layer

    mean_all = 0
    std_all = 0

    mean_all_incorrect_labels = 0
    std_all_incorrect_labels = 0

    for col_index in range(2):
        for row_index in range(2):

            indices_correct = np.where((targets == col_index) & (targets == y_predicted_on_layer))

            indices = indices_correct

            if row_index == col_index:
                #print('mean:', np.mean(matrix[indices, row_index][0]))
                #print('std:', np.std(matrix[indices, row_index][0]))
                mean_all += np.mean(matrix[indices, row_index][0])
                std_all += np.std(matrix[indices, row_index][0])
            else:
                mean_all_incorrect_labels += np.mean(matrix[indices, row_index][0])
                std_all_incorrect_labels += np.std(matrix[indices, row_index][0])
    mean_all /= 2
    std_all /= 2

    print("Averaged mean: ", mean_all)
    print("Averaged std: ", std_all)

    mean_all_incorrect_labels /= 2 * (2 - 1)
    std_all_incorrect_labels /= 2 * (2 - 1)

    print("Averaged mean_all_incorrect_labels: ", mean_all_incorrect_labels)
    print("Averaged std_all_incorrect_labels: ", std_all_incorrect_labels)
    return mean_all, std_all



def analysis_val_set(model, inputs, targets,size):
    num_layers = len(model.layers)
    # val set
    num_val_samples = size
    test_data_record_indices = range(0, num_val_samples)

    batch_size = size
    num_batches = int(num_val_samples / batch_size)
    chunk_indices_validation = np.array_split(test_data_record_indices, num_batches)
    y_predicted_on_layer = np.zeros((num_layers, num_val_samples))
    cumulative_goodness_on_layer = np.zeros((num_layers, num_val_samples))
    softmax_output_on_layer = np.zeros((num_layers, num_val_samples, 2))  # 2 is the number of softmax neurons

    for i in range(num_batches):
        x_ = inputs[chunk_indices_validation[i]]

        temp_y_predicted_on_layer, temp_cumulative_goodness_on_layer, temp_softmax_output_on_layer = \
            model.light_predict_analysis(x=x_, num_layers=num_layers)

        y_predicted_on_layer[:, chunk_indices_validation[i]] = temp_y_predicted_on_layer#.detach().cpu().numpy()

        cumulative_goodness_on_layer[:, chunk_indices_validation[i]] = \
            temp_cumulative_goodness_on_layer#.detach().cpu().numpy()

        softmax_output_on_layer[:, chunk_indices_validation[i], :] = temp_softmax_output_on_layer

    mean = []
    std =  []
    # t = targets.detach().cpu().numpy()
    for i in range(num_layers):
        temp_mean, temp_std = calculate_goodness_distributions(softmax_output_on_layer[i, :, :], y_predicted_on_layer[i, :], targets.detach().cpu().numpy(), num_layers)
        mean.append(temp_mean)
        std.append(temp_std)

    return mean, std
