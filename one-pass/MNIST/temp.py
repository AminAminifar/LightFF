import numpy as np
import  torch

a = [1,2,3,4,5]
d = 1
print(sum(a[1:d + 2]))

from itertools import islice

obj = [10, 20, 30, 40, 50]
start = 0
stop = 1

for i, item in islice(enumerate(obj), start, stop):
    print(f"Index: {i}, Item: {item}")

a = np.array([1, 2, 3, 4, 5])
b = np.array([10, 3, 2, 4, 2])
print(np.where(a == b))
print(np.arange(2))
print(b[np.arange(2)])


aa = np.array([[1,23,3],[1,1,1]])
print(aa.shape[0] )

# print("i,j>>>>", i, j)
# print(type([h.pow(2).mean(1)]), len([h.pow(2).mean(1)]))
# print(type(h.pow(2).mean(1)))
# print(">", h.pow(2).mean(1).size())
# print(type(h))
# print(h.size())
# print(sum([h.pow(2).mean(1)]).unsqueeze(1).size())

# print(type(h.pow(2).mean(1)))
# print(h.pow(2).mean(1).size())
# print(cumulative_goodness_on_layer[j, :])
# print(cumulative_goodness_on_layer[j, :].shape)

aaa = torch.randint(10, (2, 2)).byte().float()
print(aaa)
print(aaa.pow(2))
print(aaa.mean(1))
# print(aaa.mean(0))

# print(">>>", self.forward(x_pos).size())
# print(">>>", self.forward(x_pos).pow(2).mean(1).size())
# print(">>>", self.forward(x_pos).size()[1])
# exit()

# print(">>>",cumulative_goodness_on_layer[col_index, indices].shape, cumulative_goodness_on_layer[col_index, indices], indices.shape)  # len(indices)