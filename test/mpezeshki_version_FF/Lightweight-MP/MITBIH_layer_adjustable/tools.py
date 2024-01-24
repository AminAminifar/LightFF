import numpy as np
import torch
from Evaluation import print_results


def calculate_goodness_distributions(matrix, targets):

    mean_all = 0
    std_all = 0

    for col_index in range(5):  # 10
        for row_index in range(5):  # 10
            indices = np.where(targets == col_index)

            #temp = matrix[indices, row_index][0]
            #mean_value = np.mean(matrix[indices, row_index][0])

            if row_index == col_index:
                #print('mean:', np.mean(matrix[indices, row_index][0]))
                #print('std:', np.std(matrix[indices, row_index][0]))
                mean_all += np.mean(matrix[indices, row_index][0])
                std_all += np.std(matrix[indices, row_index][0])
    mean_all /= 5
    std_all /= 5
    print("Averaged mean: ", mean_all)
    print("Averaged std: ", std_all)
    return mean_all,std_all


def analysis_val_set(model, inputs, targets,length_network):
    # test set
    num_test_samples = 10000
    test_data_record_indices = range(0, num_test_samples)

    batch_size = 1000
    num_batches = int(num_test_samples / batch_size)
    chunk_indices_validation = np.array_split(test_data_record_indices, num_batches)
    y_predicted = np.zeros((length_network,num_test_samples))
    goodness_per_label = np.zeros((length_network,num_test_samples, 5))

    for i in range(num_batches):
        x_pos_ = inputs[chunk_indices_validation[i]]

        temp, temp_goodness = model.light_predict_4l(x_pos_,length_network)
        y_predicted[:,chunk_indices_validation[i]] = temp[:,:].detach().cpu().numpy()

        for j in range(length_network):
            goodness_per_label[j,chunk_indices_validation[i],:] = temp_goodness[j].detach().cpu().numpy()

    for i in range(length_network):
        print("\nResults for the {}VALIDATION{} set for {}Layer{} ".
            format('\033[1m', '\033[0m', '\033[1m', '\033[0m')+ str(i+1))
        print_results(targets.detach().cpu().numpy(), y_predicted[i])
        print('\tError:', 1.0 - torch.eq(torch.tensor(y_predicted[i]), targets.detach().cpu()).float().mean().item())


    mean = []
    std =  []
    # t = targets.detach().cpu().numpy()
    for i in range(length_network):
        temp_mean, temp_std = calculate_goodness_distributions(goodness_per_label[i], targets.detach().cpu().numpy())
        mean.append(temp_mean)
        std.append(temp_std)

    return mean, std