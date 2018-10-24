import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def pre_process(file):
    """
    Pre-process the iris data.

    Args:
        file - the iris data file
    Returns:
        a pre-processed training dataframe, a pre-processed 
        test dataframe
    """

    # read in the iris data
    data = pd.read_csv(file)

    # split into training and test sets
    train, test = train_test_split(data, test_size=0.2)

    # pre-process the training features and labels
    x_train = train[['f1','f2','f3','f4']]
    y_train = pd.get_dummies(data['species'])

    # merge things back together
    train_out = pd.DataFrame(x_train, columns=['f1','f2','f3','f4'])
    train_out = train_out.join(y_train)

    return train_out, test

class NNModel(torch.nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):

        # call class constructor
        super(NNModel, self).__init__()

        # use the nn package to create the layers of our network
        # (i.e., our architecture). A first fully connected layer
        # (fc1), a hidden layer (defined by the output/input sizes
        # of our FC layers), and a second fully connected layer (fc2).
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, num_classes) 

        # also use the nn package to define our activation function.
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # We pass the input through the full connected layers and
        # apply our activation function.
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        return out

def nn_fit(x, y, input_size, hidden_size, num_classes, learning_rate, epochs):
    """
    Train a logistic regresson model using SGD and pytorch. 

    Args:
        x - feature array, a numpy array
        y - response array, a numpy array
        input_size - the number of features in the input data
        hidden_size - size of the hidden layer
        num_classes - number of unique labels to output
        learning_rate - learning rate used in SGD
        epochs - number of epochs for the SGD loop
    Returns:
        The trained model
    """

    # initialize the model
    model = NNModel(input_size, hidden_size, num_classes)

    # our error/loss function
    criterion = nn.BCEWithLogitsLoss()

    # define our SGD optimizer
    optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # loop over our epochs, similar to our previous implementations
    for epoch in range(epochs):

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

    # import and pre-process the iris data
    train, test = pre_process('../data/iris.csv')

    # train our model
    model = nn_fit(train[['f1','f2','f3','f4']].values, 
            train[['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']].values, 
            4, 5, 3, 0.1, 10000)

    # make predictions on our test data
    X_test = Variable(torch.from_numpy(test[['f1','f2','f3','f4']].values).float())
    out = model(X_test)
    _, labels = torch.max(out.data, 1)

    # Parse the results
    species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    predictions = []
    for label in labels:
        predictions.append(species[label])

    # Calculate accuracy
    acc = accuracy_score(test['species'].values, predictions)
    print('Accuracy: ', acc)

if __name__ == "__main__":
    main()
