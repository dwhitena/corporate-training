import math

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error

class PyTorchLRModel(torch.nn.Module):

    def __init__(self, input_dim, output_dim):

        # call class constructor
        super(PyTorchLRModel, self).__init__()
        
        # use the nn package to create a linear layer
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):

        # Define the "forward" pass of this model. Think of this
        # for now as just the method that takes data input and
        # passes this through the model to create output (i.e., a prediction).
        out = self.linear(x)
        return out

def pytorch_lr_fit(x, y, learning_rate, epochs):
    """
    Train a (potentially multiple) linear regresison model 
    using SGD and pytorch.

    Args:
        x - feature array, a numpy array
        y - response array, a numpy array
        learning_rate - learning rate used in SGD
        epochs - number of epochs for the SGD loop
    Returns:
        The trained model
    """

    # define the number of features that we expect as input
    # (in input_dimension), and the number of output features
    # (in output_dimension). 
    input_dimension = x.ndim
    output_dimension = y.ndim

    # prep the shapes of x and y for pytorch
    if input_dimension == 1:
        x = x[:, np.newaxis]
    else:
        input_dimension = x.shape[1]
    if output_dimension == 1:
        y = y[:, np.newaxis]
    else:
        output_dimension = y.shape[1]

    # initialize the model
    model = PyTorchLRModel(input_dimension, output_dimension)
    
    # our error/loss function
    criterion = torch.nn.MSELoss()
    
    # define our SGD optimizer
    optimiser = torch.optim.SGD(model.parameters(), lr = learning_rate) 

    # loop over our epochs, similar to our previous implementation
    for epoch in range(epochs):

        # increment the epoch count
        epoch +=1
        
        # define our feature and response variables
        features = Variable(torch.from_numpy(x).float())
        response = Variable(torch.from_numpy(y).float())
        
        #clear the gradients
        optimiser.zero_grad()
        
        # calculate the predicted values
        predictions = model.forward(features)
        
        # calculate our loss
        loss = criterion(predictions, response)
        
        # implement our gradient-based updates to our
        # parammeters (putting them "back" into the model
        # via a "backward" update)
        loss.backward()
        optimiser.step()

    return model

def main():
    
    # import the training data
    training_data = pd.read_csv('../data/training.csv')
    training_data['bmi_sq'] = training_data['bmi'].apply(lambda x: x ** 2)
    training_data['ltg_sq'] = training_data['ltg'].apply(lambda x: x ** 2)

    # scale the features and response
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(training_data[['bmi', 'bmi_sq', 'ltg', 'ltg_sq', 'y']])

    # fit our model
    model = pytorch_lr_fit(np.array(train_data[:, 0:4]), np.array(train_data[:, 4]), 0.1, 1000)
    
    # read in and pre-process our test data
    test_data = pd.read_csv('../data/test.csv')
    test_data['bmi_sq'] = test_data['bmi'].apply(lambda x: x ** 2)
    test_data['ltg_sq'] = test_data['ltg'].apply(lambda x: x ** 2)
    test_data = scaler.transform(test_data[['bmi', 'bmi_sq', 'ltg', 'ltg_sq', 'y']])

    # test input for the model and observation tensors
    test_input = Variable(torch.from_numpy(test_data[:, 0:4]).float())
    
    # make our predictions by running the test input "forward"
    # through the model
    predictions = model(test_input)

    # calculate our RMSE
    rmse = math.sqrt(mean_squared_error(predictions.data.numpy(), test_data[:, 4]))
    print('RMSE: %0.4f'% rmse)

if __name__ == "__main__":
    main()

