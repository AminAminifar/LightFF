import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from Evaluation import print_results


def plot_softmax_distributions(matrix, y_predicted_on_layer, targets, num_layers):  # matrix is softmax_output_on_layer
    column_titles = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4',
                     'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9']

    row_titles = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4',
                  'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9']

    mean_all = 0
    std_all = 0

    mean_all_incorrect_labels = 0
    std_all_incorrect_labels = 0

    fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(15, 15))
    for col_index in range(10):
        for row_index in range(10):
            indices_all = np.where(targets == col_index)
            indices_correct = np.where((targets == col_index) & (targets == y_predicted_on_layer))
            indices_incorrect = np.where((targets == col_index) & (targets != y_predicted_on_layer))

            indices = indices_correct

            sns.histplot(data=matrix[indices, row_index][0], ax=axes[row_index, col_index], kde=True, legend=False,
                         bins=50)
            mean_value = np.mean(matrix[indices, row_index][0])
            axes[row_index, col_index].axvline(mean_value, color='red', linestyle='dashed', linewidth=1, label='Mean')

            if row_index == col_index:
                print('mean:', np.mean(matrix[indices, row_index][0]))
                print('std:', np.std(matrix[indices, row_index][0]))
                mean_all += np.mean(matrix[indices, row_index][0])
                std_all += np.std(matrix[indices, row_index][0])
            else:
                mean_all_incorrect_labels += np.mean(matrix[indices, row_index][0])
                std_all_incorrect_labels += np.std(matrix[indices, row_index][0])
    mean_all /= 10
    std_all /= 10

    print("Averaged mean: ", mean_all)
    print("Averaged std: ", std_all)

    mean_all_incorrect_labels /= 10 * (10 - 1)
    std_all_incorrect_labels /= 10 * (10 - 1)

    print("Averaged mean_all_incorrect_labels: ", mean_all_incorrect_labels)
    print("Averaged std_all_incorrect_labels: ", std_all_incorrect_labels)

    plt.tight_layout()
    plt.show()


def plot_goodness_distributions(cumulative_goodness_on_layer, y_predicted_on_layer, targets, num_layers):
    """ we plot the goodness results to analyze their changes
    for correct classification and misclassification cases
    and to see if there is a pattern  """

    row_titles = ['Correct Classification Cases', 'Misclassification Cases', 'All Cases']
    column_titles = ['Up to Layer 1', 'Up to Layer 2', 'Up to Layer 3', 'Up to Layer 4', 'Up to Layer 5']



    num_rows = 3  # one row for correct classification cases, one row for misclassification cases, and one for all
    num_cols = num_layers  # to see the pattern in each layer

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(10, 10))
    # Iterate through each column and plot the distribution using Seaborn
    for col_index in range(num_cols):  # 10
        for row_index in range(num_rows):  # 10
            if row_index == 0:  # Correct Classification Cases
                # indices_correct_classification = np.where(targets == y_predicted_on_layer[col_index, :])
                indices = np.where(targets == y_predicted_on_layer[col_index, :])
                # print("Num correct predictions in validation set batch: ")
            elif row_index == 1:  # Misclassification Cases
                # indices_misclassification = np.where(targets != y_predicted_on_layer[col_index, :])
                indices = np.where(targets != y_predicted_on_layer[col_index, :])
                # print("Num incorrect predictions in validation set batch: ")
            elif row_index == 2:  # All Cases
                num_samples = cumulative_goodness_on_layer.shape[1]
                indices = np.arange(num_samples)
                # print("Num predictions in validation set batch: ")
            indices = np.array(indices).flatten()
            # print(len(indices))


            sns.histplot(data=cumulative_goodness_on_layer[col_index, indices], ax=axes[row_index, col_index],
                         kde=True, legend=False, bins=50)  # cumulative_goodness_on_layer[col_index, indices][0]
            # axes[row_index, col_index].set_title(column_titles[col_index])
            # Add a vertical line at the mean
            mean_value = np.mean(cumulative_goodness_on_layer[col_index, indices])  # [0]
            axes[row_index, col_index].axvline(mean_value, color='red', linestyle='dashed', linewidth=1, label='Mean')

            print("\n", row_titles[row_index], column_titles[col_index])
            print('mean:', np.mean(cumulative_goodness_on_layer[col_index, indices]))  # [0]
            print('std:', np.std(cumulative_goodness_on_layer[col_index, indices]))  # [0]

    plt.tight_layout()
    plt.savefig('dist.pdf', bbox_inches='tight')
    plt.show()


def analysis_val_set(model, inputs, targets):
    num_layers = len(model.layers)
    # val set
    num_val_samples = 10000
    test_data_record_indices = range(0, num_val_samples)

    batch_size = 1000
    num_batches = int(num_val_samples / batch_size)
    chunk_indices_validation = np.array_split(test_data_record_indices, num_batches)
    y_predicted_on_layer = np.zeros((num_layers, num_val_samples))
    cumulative_goodness_on_layer = np.zeros((num_layers, num_val_samples))
    softmax_output_on_layer = np.zeros((num_layers, num_val_samples, 10))  # 10 is the number of softmax neurons

    for i in range(num_batches):
        x_ = inputs[chunk_indices_validation[i]]

        temp_y_predicted_on_layer, temp_cumulative_goodness_on_layer, temp_softmax_output_on_layer = \
            model.light_predict_analysis(x=x_, num_layers=num_layers)

        y_predicted_on_layer[:, chunk_indices_validation[i]] = temp_y_predicted_on_layer#.detach().cpu().numpy()

        cumulative_goodness_on_layer[:, chunk_indices_validation[i]] = \
            temp_cumulative_goodness_on_layer#.detach().cpu().numpy()

        softmax_output_on_layer[:, chunk_indices_validation[i], :] = temp_softmax_output_on_layer

    plot_softmax_distributions(softmax_output_on_layer[3, :, :], y_predicted_on_layer[0, :],
                               targets.detach().cpu().numpy(), num_layers)

    exit()

    for i in range(num_layers):

        print("\nResults for the {}VALIDATION{} set for {}Layer{} ".
              format('\033[1m', '\033[0m', '\033[1m', '\033[0m'), i + 1, ":")
        print_results(targets.detach().cpu().numpy(), y_predicted_on_layer[i, :])
        print('\tError:', 1.0 - torch.eq(torch.tensor(y_predicted_on_layer[i, :]),
                                         targets.detach().cpu()).float().mean().item())

    plot_goodness_distributions(cumulative_goodness_on_layer, y_predicted_on_layer,
                                targets.detach().cpu().numpy(), num_layers)

