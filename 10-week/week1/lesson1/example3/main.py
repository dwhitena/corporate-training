import numpy as np
import torch

# define the url where we will get the iris data set CSV
url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# generate a numpy array from the CSV file
data = np.genfromtxt(url, delimiter=",")

# create a tensor from the numpy array with PyTorch
data_tensor = torch.from_numpy(data)

# print the tensor size 
print("\nsize: ", data_tensor.size())

# similar to numpy, we can slice and dice the tensor
print("\nrow: ", data_tensor[1, :])
