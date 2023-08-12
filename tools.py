import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_goodness_distributions(matrix, targets):

    column_titles = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4',
                     'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9']

    row_titles = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4',
                  'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9']

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


    # Adjust layout and display the plots

    plt.tight_layout()
    plt.show()

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

