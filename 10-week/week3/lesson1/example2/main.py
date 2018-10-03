import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def pre_process(df, threshold):
    """
    Pre-process the LendingClub loan data.

    Args:
        df - the loan data in dataframe format
    Returns:
        a pre-processed dataframe
    """

    # encode the class labels
    df['class'] = df['Interest.Rate'].apply(
            lambda x: 1.0 if float(x.replace('%', '')) <= threshold else 0.0)

    # select the minimum FICO score
    df['fico_score'] = df['FICO.Range'].apply(lambda x: float(x.split('-')[0]))

    # standardize the FICO score
    df['fico_score'] = df['fico_score'].apply(
            lambda x: (x - df['fico_score'].min())/(df['fico_score'].max() - df['fico_score'].min()))

    return df[['fico_score', 'class']]

class LogRegModel(torch.nn.Module):

    def __init__(self, input_dim, output_dim):

        # call class constructor
        super(LogRegModel, self).__init__()

        # use the nn package to create a linear layer
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):

        # First we pass the input through the linear layer (like we did before),
        # then we pass that through sigmoid, which implements the logistic
        # function.
        out = torch.sigmoid(self.linear(x)) 
        return out

def log_reg_fit(x, y, learning_rate, epochs):
    """
    Train a logistic regresson model using SGD and pytorch. 

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
    model = LogRegModel(input_dimension, output_dimension)

    # our error/loss function
    criterion = torch.nn.BCELoss()

    # define our SGD optimizer
    optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # loop over our epochs, similar to our previous implementation
    for epoch in range(epochs):

        # increment the epoch count
        epoch +=1

        # define our feature and response variables
        features = Variable(torch.from_numpy(x).float())
        labels = Variable(torch.from_numpy(y).float())

        # clear the gradients
        optimiser.zero_grad()

        # calculate the predicted values
        predictions = model.forward(features)

        # calculate our loss
        loss = criterion(predictions, labels)

        # implement our gradient-based updates to our
        # parammeters (putting them "back" into the model
        # via a "backward" update)
        loss.backward()
        optimiser.step()

    return model

def main():

    # import the loan data
    df = pd.read_csv('../data/loan_data.csv')

    # pre-process the loan data
    df = pre_process(df, 12.0)
    
    # split the data into training and test sets
    train, test = train_test_split(df, test_size=0.2)

    # train our model
    model = log_reg_fit(train['fico_score'].values, train['class'].values, 0.1, 10000)

    # make predictions on our test data
    raw_predictions = model(Variable(torch.from_numpy(test['fico_score'].values[:, np.newaxis]).float()))
    predictions = []
    for prediction in raw_predictions:
        if prediction.data.numpy()[0] > 0.50:
            predictions.append(1.0)
        else:
            predictions.append(0.0)

    # calculate our accuracy
    acc = accuracy_score(test['class'].values, predictions)
    print('Accuracy: ', acc)

if __name__ == "__main__":
    main()
