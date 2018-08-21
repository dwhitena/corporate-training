from math import sqrt

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from sklearn import linear_model
import torch
from torch.autograd import Variable

def squared_error(prediction, observation):
    """
    Calculates the squared error.

    Args:
        prediction - the prediction from our linear regression model
        observation - the observed data point
    Returns:
        The squared error
    """
    return (observation - prediction) ** 2

def ols_fit(x, y):
    """
    Calculates the intercept and slope parameters using OLS for
    a linear regression model.

    Args:
        x - feature array
        y - response array
    Returns:
        The intercept and slope parameters
    """
    
    # calculate the mean of x and y
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # Using the derived OLS formula to calculate
    # the intercept and slope.
    numerator = 0
    denominator = 0
    for i in range(len(x)):
        numerator += (x[i] - mean_x) * (y[i] - mean_y)
        denominator += (x[i] - mean_x) ** 2
    slope = numerator / denominator
    intercept= mean_y - (slope * mean_x)

    return intercept, slope

def sgd_fit(x, y, learning_rate, epochs):
    """
    Calculates the intercept and slope parameters using SGD for
    a linear regression model.

    Args:
        x - feature array
        y - response array
        learning_rate - learning rate
        epochs - the number of epochs to use in the SGD loop
    Returns:
        The intercept and slope parameters and the sum of
        squared error for the last epoch
    """

    # initialize the slope and intercept
    slope = 0.0
    intercept = 0.0

    # set the number of observations in the data
    N = float(len(y))

    # loop over the number of epochs
    for i in range(epochs):

        # calculate our current predictions
        predictions = (slope * x) + intercept
        
        # calculate the sum of squared errors for this epoch
        error = sum([data**2 for data in (y-predictions)]) / N

        # calculate the gradients for the slope and intercept
        slope_gradient = -(2/N) * sum(x * (y - predictions))
        intercept_gradient = -(2/N) * sum(y - predictions)
        
        # update the slope and intercept
        slope = slope - (learning_rate * slope_gradient)
        intercept = intercept - (learning_rate * intercept_gradient)

    return intercept, slope, error

def sm_ols_fit(x, y):
    """
    Calculates the intercept and slope parameters using OLS for
    a linear regression model, with statsmodels.

    Args:
        x - feature array
        y - response array
    Returns:
        The intercept and slope parameters
    """

    # add a constant column to the x values which
    # represents the intercept
    x = sm.add_constant(x)

    # define the OLS model
    model = sm.OLS(y, x)

    # train the model
    results = model.fit()

    return results.params[0], results.params[1]

def sklearn_ols_fit(x, y):
    """
    Calculates the intercept and slope parameters using OLS for
    a linear regression model, with scikit-learn.

    Args:
        x - feature array
        y - response array
    Returns:
        The intercept and slope parameters
    """

    # define the model
    lr = linear_model.LinearRegression()
    
    # train the model
    lr.fit(x[:, np.newaxis], y)

    return lr.intercept_, lr.coef_[0]

def sklearn_sgd_fit(x, y):
    """
    Calculates the intercept and slope parameters using SGD for
    a linear regression model, with scikit-learn.

    Args:
        x - feature array
        y - response array
    Returns:
        The intercept and slope parameters
    """

    # define the model
    lr = linear_model.SGDRegressor(max_iter=1000)

    # traing the model
    lr.fit(x[:, np.newaxis], y)

    return lr.intercept_[0], lr.coef_[0]

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

def pytorch_sgd_fit(x, y, learning_rate, epochs):
    """
    Calculates the intercept and slope parameters using SGD for
    a linear regression model, with pytorch.

    Args:
        x - feature array
        y - response array
        learning_rate - learning rate used in SGD
        epochs - number of epochs for the SGD loop
    Returns:
        The intercept and slope parameters
    """

    # create the model using only one "node", which will
    # correspond to our single linear regression model
    input_dimension = 1
    output_dimension = 1

    # define the model
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
        features = Variable(torch.from_numpy(x[:, np.newaxis]).float())
        response = Variable(torch.from_numpy(y[:, np.newaxis]).float())
        
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

    # extract the model parameters to return
    params = []
    for param in model.parameters():
        params.append(param.data[0])

    return params[1].item(), params[0][0].item()

def main():
    
    # import the data
    data = pd.read_csv('../data/Advertising.csv')
    
    # scale the feature and response
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[['TV', 'Sales']])
    
    # fit our model using our various implementations
    int_ols, slope_ols = ols_fit(data_scaled[:, 0], data_scaled[:, 1])
    int_sgd, slope_sgd, _ = sgd_fit(data_scaled[:, 0], data_scaled[:, 1], 0.1, 1000)
    int_sm_ols, slope_sm_ols = sm_ols_fit(data_scaled[:, 0], data_scaled[:, 1])
    int_sk_ols, slope_sk_ols = sklearn_ols_fit(data_scaled[:, 0], data_scaled[:, 1])
    int_sk_sgd, slope_sk_sgd = sklearn_sgd_fit(data_scaled[:, 0], data_scaled[:, 1])
    int_pt_sgd, slope_pt_sgd = pytorch_sgd_fit(data_scaled[:, 0], data_scaled[:, 1], 0.1, 1000)

    # output the results
    delim = "-----------------------------------------------------------------"
    print("\nOLS\n{delim}\n intercept: {intercept}, slope: {slope}"
            .format(delim=delim, intercept=int_ols, slope=slope_ols))
    print("\nSGD\n{delim}\n intercept: {intercept}, slope: {slope}"
            .format(delim=delim, intercept=int_sgd, slope=slope_sgd))
    print("\nstatsmodels OLS\n{delim}\n intercept: {intercept}, slope: {slope}"
            .format(delim=delim, intercept=int_sm_ols, slope=slope_sm_ols))
    print("\nsklearn OLS\n{delim}\n intercept: {intercept}, slope: {slope}"
            .format(delim=delim, intercept=int_sk_ols, slope=slope_sk_ols))
    print("\nsklearn SGD\n{delim}\n intercept: {intercept}, slope: {slope}"
            .format(delim=delim, intercept=int_sk_sgd, slope=slope_sk_sgd))
    print("\npytorch SGD\n{delim}\n intercept: {intercept}, slope: {slope}"
            .format(delim=delim, intercept=int_pt_sgd, slope=slope_pt_sgd))

if __name__ == "__main__":
    main()

