import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from sklearn.metrics import f1_score, accuracy_score
import mitbih_dataset.load_data as bih 
import numpy as np
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import CHBMIT.chbmit_dataset.load_data as chb
import torch.utils.data as Data

print('CHBMIT_PEPITA')
layers = [1024, 2000,2000,2000,2000]
classes = 2
epoch_set = 100
length_network = len(layers)-1
print('layers: ' + str(length_network))
print(layers)

# models with Dropout

class One_Pass(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.softmax_l = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.softmax_l.weight)
        self.softmax = torch.nn.Softmax(dim=1)
        self.opt = Adam(self.parameters(), lr=0.03)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        #  x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        output_l = self.softmax_l(x)
        output = self.softmax(output_l)  # .argmax(1)
        return output_l, output

    def train(self, x, y):
        self.opt.zero_grad()
        yhat,_ = self.forward(x)
        # y_one_hot = nn.functional.one_hot(y, num_classes=10).to(torch.float32)
        loss = self.criterion(yhat, y)
        loss.backward()
        self.opt.step()
        
class NetFC1x1024DOcust(nn.Module):
    def __init__(self,dims,classes):
        super().__init__()
        #self.layers = []
        self.layers = torch.nn.ModuleDict(
            {f"fc{d+1}": nn.Linear(dims[d],dims[d+1],bias=False) for d in range(len(dims)-1)}
            )
        
        for d in range(len(dims)-1):
            #self.layers+=[nn.Linear(dims[d],dims[d+1],bias=False)]
            nin = dims[d]
            limit = np.sqrt(6.0 / nin)
            torch.nn.init.uniform_(self.layers[f"fc{d+1}"].weight, a=-limit, b=limit)

        self.fc_last = nn.Linear(dims[-1], classes,bias=False)
        fc_last_nin = dims[-1]
        fc_last_limit = np.sqrt(6.0 / fc_last_nin)
        torch.nn.init.uniform_(self.fc_last.weight, a=-fc_last_limit, b=fc_last_limit)

        
    def forward(self, x, do_masks):
        x = F.relu(self.layers[f"fc{1}"](x))
        # apply dropout --> we use a custom dropout implementation because we need to present the same dropout mask in the two forward passes
        if do_masks is not None:

            for i in range(1,len(self.layers)):
                x = x * do_masks[i-1]
                x = F.relu(self.layers[f"fc{i+1}"](x))
            x = x * do_masks[i]
            x = F.softmax(self.fc_last(x),dim=1)

        else:
            for i in range(1,len(self.layers)):
                x = F.relu(self.layers[f"fc{i+1}"](x))
            x = F.softmax(self.fc_last(x),dim=1)
        return x
    

def calculate_goodness_distributions(matrix, y_predicted_on_layer, targets):  # matrix is softmax_output_on_layer

    mean_all = 0
    std_all = 0

    mean_all_incorrect_labels = 0
    std_all_incorrect_labels = 0

    for col_index in range(classes):
        for row_index in range(classes):

            indices_correct = np.where((targets == col_index) & (targets == y_predicted_on_layer))

            indices = indices_correct

            if row_index == col_index:

                mean_all += np.mean(matrix[indices, row_index][0])
                std_all += np.std(matrix[indices, row_index][0])
            else:
                mean_all_incorrect_labels += np.mean(matrix[indices, row_index][0])
                std_all_incorrect_labels += np.std(matrix[indices, row_index][0])
    mean_all /= classes
    std_all /= classes

    print("Averaged mean: ", mean_all)
    print("Averaged std: ", std_all)

    return mean_all, std_all


def each_patient_detection(which_patients,layers,length_network,classes,epoch_set):

    # set hyperparameters
    ## learning rate
    eta = 0.001  
    ## dropout keep rate
    keep_rate = 0.9
    ## loss --> used to monitor performance, but not for parameter updates (PEPITA does not backpropagate the loss)
    criterion = nn.CrossEntropyLoss()
    ## optimizer (choose 'SGD' o 'mom')
    optim = 'mom' # --> default in the paper
    if optim == 'SGD':
        gamma = 0
    elif optim == 'mom':
        gamma = 0.9
    ## batch size
    batch_size = 4 # --> default in the paper

    torch.manual_seed(1234)

    X_train,Y_train,X_val,Y_val,X_test,Y_test = chb.load_data(which_patients)

    train_size =X_train.shape[0]
    val_size =X_val.shape[0]
    test_size =X_test.shape[0]

    X_train = X_train[:,::2]
    X_test = X_test[:,::2]
    X_val = X_val[:,::2]

    X_train = torch.tensor(X_train).to(torch.float32).pow(2) # pow 2 better
    Y_train = torch.tensor(Y_train).to(torch.int64)
    X_val = torch.tensor(X_val).to(torch.float32).pow(2)

    Y_val = torch.tensor(Y_val).to(torch.int64)
    X_test = torch.tensor(X_test).to(torch.float32).pow(2)
    Y_test = torch.tensor(Y_test).to(torch.int64)

    trainset = Data.TensorDataset(X_train,Y_train)
    valset = Data.TensorDataset(X_val,Y_val)
    testset = Data.TensorDataset(X_test,Y_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True)

    valloader = torch.utils.data.DataLoader(valset, batch_size=val_size,
                                            shuffle=True)

    testloader = torch.utils.data.DataLoader(testset, batch_size=test_size,
                                            shuffle=False)


    net = NetFC1x1024DOcust(layers,classes)


    # define function to register the activations --> we need this to compare the activations in the two forward passes
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    for name, layer in net.named_modules():
        #print(name,'---',layer)
        layer.register_forward_hook(get_activation(name))

    # define B --> this is the F projection matrix in the paper (here named B because F is torch.nn.functional)
    nin = layers[0]
    sd = np.sqrt(6/nin)
    B = (torch.rand(nin,classes)*2*sd-sd)*0.05  # B is initialized with the He uniform initialization (like the forward weights)

    # do one forward pass to get the activation size needed for setting up the dropout masks
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    images = torch.flatten(images, 1) # flatten all dimensions except batch        
    outputs = net(images,do_masks=None)
    layers_act = []
    for key in activation:
        if 'fc' in key or 'conv' in key:
            layers_act.append(F.relu(activation[key]))
            
    # set up for momentum
    if optim == 'mom':
        gamma = 0.9
        v_w_all = []
        for l_idx,w in enumerate(net.parameters()):
            #print(l_idx,'---',w.size())
            if len(w.shape)>1:
                with torch.no_grad():
                    v_w_all.append(torch.zeros(w.shape))

    # Train and test the model
    test_accs = []
    for epoch in range(epoch_set):  # loop over the dataset multiple times
        
        # learning rate decay
        if epoch in [60,90]: 
            eta = eta*0.1
            print('eta decreased to ',eta)
        
        # loop over batches
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, target = data
            inputs = torch.flatten(inputs, 1) # flatten all dimensions except batch
            target_onehot = F.one_hot(target,num_classes=classes)
            
            # create dropout mask for the two forward passes --> we need to use the same mask for the two passes
            do_masks = []
            if keep_rate < 1:
                for l in layers_act[:-1]:
                    input1 = l
                    do_mask = Variable(torch.ones(inputs.shape[0],input1.data.new(input1.data.size()).shape[1]).bernoulli_(keep_rate))/keep_rate
                    do_masks.append(do_mask)
                do_masks.append(1) # for the last layer we don't use dropout --> just set a scalar 1 (needed for when we register activation layer)
            
            # forward pass 1 with original input --> keep track of activations
            outputs = net(inputs,do_masks)
            layers_act = []
            cnt_act = 0
            for key in activation:
                if 'fc' in key or 'conv' in key:
                    layers_act.append(F.relu(activation[key])* do_masks[cnt_act]) # Note: we need to register the activations taking into account non-linearity and dropout mask
                    #layers_act.append(F.relu(activation[key]))
                    cnt_act += 1
                    
            # compute the error
            error = outputs - target_onehot  
            
            # modify the input with the error
            error_input = error @ B.T
            mod_inputs = inputs + error_input
            
            # forward pass 2 with modified input --> keep track of modulated activations
            mod_outputs = net(mod_inputs,do_masks)
            mod_layers_act = []
            cnt_act = 0
            for key in activation:
                if 'fc' in key or 'conv' in key:
                    mod_layers_act.append(F.relu(activation[key])* do_masks[cnt_act]) # Note: we need to register the activations taking into account non-linearity and dropout mask
                    #mod_layers_act.append(F.relu(activation[key]))
                    cnt_act += 1
            mod_error = mod_outputs - target_onehot
            
            # compute the delta_w for the batch
            delta_w_all = []
            v_w = []
            for l_idx,w in enumerate(net.parameters()):
                v_w.append(torch.zeros(w.shape))
                
            for l in range(len(layers_act)):
                
                # update for the last layer
                if l == len(layers_act)-1:
                    
                    if len(layers_act)>1:
                        delta_w = -mod_error.T @ mod_layers_act[-2]
                    else:
                        delta_w = -mod_error.T @ mod_inputs
                
                # update for the first layer
                elif l == 0:
                    delta_w = -(layers_act[l] - mod_layers_act[l]).T @ mod_inputs
                
                # update for the hidden layers (not first, not last)
                elif l>0 and l<len(layers_act)-1:
                    delta_w = -(layers_act[l] - mod_layers_act[l]).T @ mod_layers_act[l-1]
                
                delta_w_all.append(delta_w)
                    
            # apply the weight change
            if optim == 'SGD':
                for l_idx,w in enumerate(net.parameters()):
                    with torch.no_grad():
                        w += eta * delta_w_all[l_idx]/batch_size # specify for which layer
                        
            elif optim == 'mom':
                for l_idx,w in enumerate(net.parameters()):
                    with torch.no_grad():
                        v_w_all[l_idx] = gamma * v_w_all[l_idx] + eta * delta_w_all[l_idx]/batch_size
                        w += v_w_all[l_idx]
                        
            
            # keep track of the loss
            loss = criterion(outputs, target)
            # print statistics
            running_loss += loss.item()
            if i%500 == 499:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0
                        
        print('Testing...')
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for test_data in testloader:
                test_images, test_labels = test_data
                test_images = torch.flatten(test_images, 1) # flatten all dimensions except batch
                # calculate outputs by running images through the network
                test_outputs = net(test_images,do_masks=None)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(test_outputs.data, 1)
                total += test_labels.size(0)
                correct += (predicted == test_labels).sum().item()

        print('Test accuracy epoch {}: {} %'.format(epoch, 100 * correct / total))
        test_accs.append(100 * correct / total)


    train_error = 1-test_accs[-1]/100

    print('Finished Training')

    print('One-Pass Training')

    softmax_layers = []

    for index in range(len(layers)-2):
        softmax_layers += [One_Pass(layers[index+1],classes)]

        for epoch in range(epoch_set):  # loop over the dataset multiple times
            
            # learning rate decay
            if epoch in [60,90]: 
                eta = eta*0.1
                print('eta decreased to ',eta)
            
            # loop over batches
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, target = data
                inputs = torch.flatten(inputs, 1) # flatten all dimensions except batch
                target_onehot = F.one_hot(target,num_classes=classes)
                
                # forward pass 1 with original input --> keep track of activations
                outputs = net(inputs,do_masks=None)
                layers_act = []
                cnt_act = 0
                for key in activation:
                    if 'fc' in key or 'conv' in key:
                        layers_act.append(F.relu(activation[key])) # Note: we need to register the activations taking into account non-linearity and dropout mask
                        cnt_act += 1
                
                softmax_layers[index].train(layers_act[index],target)


            print('Testing for one_pass_'+str(index+1))
            correct = 0
            total = 0
            # since we're not training, we don't need to calculate the gradients for our outputs
            with torch.no_grad():
                for test_data in testloader:
                    test_images, test_labels = test_data
                    test_images = torch.flatten(test_images, 1) # flatten all dimensions except batch
                    # calculate outputs by running images through the network

                    test_outputs = net(test_images,do_masks=None)
                    
                    layers_act = []
                    cnt_act = 0
                    for key in activation:
                        if 'fc' in key or 'conv' in key:
                            layers_act.append(F.relu(activation[key])) # Note: we need to register the activations taking into account non-linearity and dropout mask
                            cnt_act += 1

                    _, one_pass_1_output = softmax_layers[index](layers_act[index])

                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(one_pass_1_output.data, 1)

                    #_, predicted = torch.max(test_outputs.data, 1)
                    total += test_labels.size(0)
                    correct += (predicted == test_labels).sum().item()

            print('Test accuracy epoch {}: {} %'.format(epoch, 100 * correct / total))
            test_accs.append(100 * correct / total)


    # calculate the threshold

    one_pass_before_softmax_all=[]
    predicted_labels_all=[]


    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for val_data in valloader:
            val_images, val_labels = val_data
            val_images = torch.flatten(val_images, 1) # flatten all dimensions except batch
            # calculate outputs by running images through the network

            real_labels = val_labels

            val_outputs = net(val_images,do_masks=None)
            _, predicted_last_layer = torch.max(val_outputs.data, 1)

            layers_act = []
            cnt_act = 0
            for key in activation:
                if 'fc' in key or 'conv' in key:
                    layers_act.append(F.relu(activation[key])) # Note: we need to register the activations taking into account non-linearity and dropout mask
                    cnt_act += 1

            
            for index in range(len(layers)-2):

                one_pass_1_before_softmax, one_pass_1_output = softmax_layers[index](layers_act[index])

                one_pass_before_softmax_all.append(one_pass_1_before_softmax)

                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(one_pass_1_output.data, 1)
                
                predicted_labels_all.append(predicted)


    mean = []
    std =  []
    # t = targets.detach().cpu().numpy()
    for i in range(len(layers)-2):
        temp_mean, temp_std = calculate_goodness_distributions(one_pass_before_softmax_all[i].detach().cpu().numpy(), predicted_labels_all[i].detach().cpu().numpy(), real_labels.detach().cpu().numpy())
        mean.append(temp_mean)
        std.append(temp_std)


    # inference in lightweight

    real_labels = []
    predict_labels= []
    predicted_with_layers = []

    with torch.no_grad():
        for test_data in testloader:
            test_images, test_labels = test_data
            test_images = torch.flatten(test_images, 1) # flatten all dimensions except batch
            # calculate outputs by running images through the network


            for i,(image,label) in enumerate(zip(test_images,test_labels)):

                confidence_flag = False  # if confident: True
                predicted_with_layers_up_to = 0

                real_labels.append(label.detach().cpu().numpy())

                output = net(image.unsqueeze(0),do_masks=None)
                _, predicted = torch.max(output.data, 1)

                layers_act = []
                cnt_act = 0
                for key in activation:
                    if 'fc' in key or 'conv' in key:
                        layers_act.append(F.relu(activation[key])) # Note: we need to register the activations taking into account non-linearity and dropout mask
                        cnt_act += 1

                
                for index in range(len(layers)-2):
                    if not confidence_flag:
                        predicted_with_layers_up_to +=1
                        one_pass_1_before_softmax, one_pass_1_output = softmax_layers[index](layers_act[index])

                        if torch.max(one_pass_1_before_softmax) > mean[index]-std[index]:  # then we are confident
                            confidence_flag = True
                            _, predicted = torch.max(one_pass_1_output.data, 1)

                if confidence_flag == False:
                    predicted_with_layers_up_to +=1

                predict_labels.append(predicted.detach().cpu().numpy().squeeze())
                predicted_with_layers.append(predicted_with_layers_up_to)


    def print_results(labels_vec, predictions_vec):
        f1_performance = f1_score(labels_vec, predictions_vec, average='macro')
        acc_performance = accuracy_score(labels_vec, predictions_vec)

        print("\tF1-score: ", f1_performance)
        print("\tAccuracy: ", acc_performance)

    print("\nResults for the {}VALIDATION{} set based on light inference: ".format('\033[1m', '\033[0m'))
    print_results(real_labels, predict_labels)

    real_labels = np.array(real_labels)
    predict_labels = np.array(predict_labels)

    print('\tError:', 1.0 - torch.eq(torch.tensor(predict_labels), torch.tensor(real_labels)).float().mean().item())

    test_error_light =  1.0 - torch.eq(torch.tensor(predict_labels), torch.tensor(real_labels)).float().mean().item()
    mean_number_layers = np.mean(predicted_with_layers)

    print("mean number of layers used: ", np.mean(predicted_with_layers))
    values, counts = np.unique(predicted_with_layers, return_counts=True)

    temp = []
    for i in range(1,length_network+1):
        #print(i)
        temp.append(predicted_with_layers.count(i))
        percentage = [x/test_size for x in temp]

    print("percentage for layers_up_to "+str(percentage))

    return train_error,test_error_light,mean_number_layers,percentage

def train_seizure(layers,length_network,classes,epoch_set):
    
    chb_list = range(1,25,1)
    total_error_train = 0
    total_error_test_new = 0
    total_light_number_layers = 0
    total_percentage = [ 0 for i in range(length_network) ] #[0,0,0,0]
    print("-------------begin-------------------")
    for i in chb_list:
        print("process for patient "+str(i))
        if i== 6 or i==14 or i==16: # we do not consider patient 6/14/16
            continue
        if i <10:
            which_patients = 'chb0'+str(i)
        else:
            which_patients = 'chb'+str(i)
        train_error,test_error_light,light_number_layers,percentage = each_patient_detection(which_patients,layers,length_network,classes,epoch_set)
        total_error_train += train_error
        total_error_test_new += test_error_light
        total_light_number_layers += light_number_layers
        total_percentage = [total_percentage[i] + percentage[i] for i in range(len(percentage))]

    print("-------------over-------------------")
    print("the mean error for train is "+str(total_error_train/21))
    print("the mean error for test_light is "+str(total_error_test_new/21))
    print("the mean number of used layers is "+str(total_light_number_layers/21))
    print("the mean percentage of used layers is "+str([x/21 for x in total_percentage]))

train_seizure(layers,length_network,classes,epoch_set)