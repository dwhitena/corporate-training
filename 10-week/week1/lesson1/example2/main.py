import numpy as np

# define the url where we will get the iris data set CSV
url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# generate a numpy array from the CSV file
data = np.genfromtxt(url, delimiter=",")

# print out the "shape" of data
print("\nshape: ", data.shape)

# print out the first and second row of data
print("\nrow 1: ", data[0,:])
print("row 2: ", data[1,:])
