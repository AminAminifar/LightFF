import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from Evaluation import print_results


def plot_goodness_distributions(matrix, targets, name='temp.pdf'):

    column_titles = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4',
                     'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9']

    row_titles = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4',
                  'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9']
    mean_all = 0
    std_all = 0

    # Create a 10 by 4 grid of subplots
    fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(15, 15))  # 10
    # Iterate through each column and plot the distribution using Seaborn
    for col_index in range(10):  # 10
        for row_index in range(10):  # 10
            indices = np.where(targets == col_index)

            sns.histplot(data=matrix[indices, row_index][0], ax=axes[row_index, col_index], kde=True, legend=False, bins=50)  # , ax=axes[row_index, col_index]
            # axes[row_index, col_index].set_title(column_titles[col_index])
            # Add a vertical line at the mean
            mean_value = np.mean(matrix[indices, row_index][0])
            axes[row_index, col_index].axvline(mean_value, color='red', linestyle='dashed', linewidth=1, label='Mean')
            if row_index == col_index:
                # print('mean:', np.mean(matrix[indices, row_index][0]))
                # print('std:', np.std(matrix[indices, row_index][0]))
                mean_all += np.mean(matrix[indices, row_index][0])
                std_all += np.std(matrix[indices, row_index][0])
    mean_all /= 10
    std_all /= 10
    print("Averaged mean: ", mean_all)
    print("Averaged std: ", std_all)



    # Adjust layout and display the plots

    plt.tight_layout()
    plt.savefig(name, bbox_inches='tight')
    # plt.show()

# def plot_goodness_distributions(matrix1, matrix2, matrix3, matrix4):
#     # Assuming you have your matrices as NumPy arrays
#     # If not, you can convert them using: np.array(your_matrix)
#
#     # Replace these with your actual data matrices
#     # matrix1 = np.random.randn(10, 4)
#     # matrix2 = np.random.randn(10, 4)
#     # matrix3 = np.random.randn(10, 4)
#     # matrix4 = np.random.randn(10, 4)
#
#
#     data_matrices = [matrix1, matrix2, matrix3, matrix4]
#     column_titles = ['Column 1', 'Column 2', 'Column 3', 'Column 4']
#
#     # Create a 10 by 4 grid of subplots
#     fig, axes = plt.subplots(nrows=10, ncols=4, figsize=(15, 20))
#
#     # Iterate through each column and plot the distribution using Seaborn
#     for col_index, data_matrix in enumerate(data_matrices):
#         for row_index in range(10):
#             sns.histplot(data_matrix[:, row_index], ax=axes[row_index, col_index], kde=True)
#             axes[row_index, col_index].set_title(column_titles[col_index])
#
#     # Adjust layout and display the plots
#     plt.tight_layout()
#     plt.show()


def analysis_val_set(model, inputs, targets):
    # test set
    num_test_samples = 5000
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
        temp_goodness_per_label_l1_l2_l3, temp_goodness_per_label_l1_l2_l3_l4 = model.light_predict_4l(x_pos_)

        y_predicted_l1[chunk_indices_validation[i]], y_predicted_l1_l2[chunk_indices_validation[i]], \
            y_predicted_l1_l2_l3[chunk_indices_validation[i]], y_predicted_l1_l2_l3_l4[chunk_indices_validation[i]] = \
            temp_l1.detach().cpu().numpy(), temp_l1_l2.detach().cpu().numpy(), \
            temp_l1_l2_l3.detach().cpu().numpy(), temp_l1_l2_l3_l4.detach().cpu().numpy()

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


    # t = targets.detach().cpu().numpy()
    print("goodness_per_label_l1: ")
    plot_goodness_distributions(goodness_per_label_l1, targets.detach().cpu().numpy(),
                                name='goodness_per_label_l1.pdf')  # goodness_per_label_l1[:1000, :], t[:1000]

    print("goodness_per_label_l1_l2: ")
    plot_goodness_distributions(goodness_per_label_l1_l2, targets.detach().cpu().numpy(),
                                name='goodness_per_label_l1_l2.pdf')

    print("goodness_per_label_l1_l2_l3: ")
    plot_goodness_distributions(goodness_per_label_l1_l2_l3, targets.detach().cpu().numpy(),
                                name='goodness_per_label_l1_l2_l3.pdf')

    print("goodness_per_label_l1_l2_l3_l4: ")
    plot_goodness_distributions(goodness_per_label_l1_l2_l3_l4, targets.detach().cpu().numpy(),
                                name='goodness_per_label_l1_l2_l3_l4.pdf')

def analysis_val_set_2l(model, inputs, targets):
    # test set
    num_test_samples = 5000
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

