import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def load_cifar10_dataset():
    """
    Download and pre-process the CIFAR10 dataset.

    Returns:
        a "loader" for our training data, a loader for our
        test data, a tuple of classes
    """

    # Create a "transform" that will allow us to transform the
    # image objects to tensors and normalize their values. Find
    # out more about PyTorch transform here: 
    # https://pytorch.org/docs/stable/torchvision/transforms.html
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # Download the traning dataset to our data folder and create a
    # data loader that will allow us to load in batches of the
    # images. torch.utils.data.DataLoader is an iterator which lets us
    # batch the data, shuffle the data, and load the data in parallel 
    #using multiprocessing workers.
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
    
    # Download the test data and create another data loader.
    testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
    
    # Create a tuple of classes contained in the data set.
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader, classes

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # A 1st convolutional layer.
        self.conv1 = nn.Conv2d(3, 6, 5)

        # A "pooling" layer. Find out more about these here:
        # http://cs231n.github.io/convolutional-networks/
        self.pool = nn.MaxPool2d(2, 2)

        # A 2nd convolutional layer.
        self.conv2 = nn.Conv2d(6, 16, 5)

        # A fully connected linear layer like we have seen before.
        self.fc1 = nn.Linear(16 * 5 * 5, 120)

        # Another fully connected linear layer.
        self.fc2 = nn.Linear(120, 84)

        # A final fully connected linear layer.
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def nn_fit(trainloader, learning_rate, epochs):
    """
    Train a CNN using SGD and pytorch. 

    Args:
        trainloader - a training data set loader
        learning_rate - learning rate used in SGD
        epochs - number of epochs for the SGD loop
    Returns:
        The trained model
    """

    # initialize the model
    model = Net()

    # our error/loss function
    criterion = nn.CrossEntropyLoss()

    # define our SGD optimizer
    optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # loop over our epochs, similar to our previous implementations
    for epoch in range(epochs):

        # This time we will keep a running total of our
        # loss so we can print out some stats.
        running_loss = 0.0
        
        # Also, as opposed to previous examples, we will be
        # uses "batches" of our data here in the gradient
        # descent training. By using the batches loaded from the data
        # loader, we can increase our training efficience (which
        # is important now that our model is much more
        # complicated). Otherwise, this is the very same thing
        # we have been doing since the first lessons.
        for i, data in enumerate(trainloader, 0):
            
            # get the inputs
            features, labels = data
            
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

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    return model

def main():

    # import and pre-process the CIFAR10 data
    trainloader, testloader, classes = load_cifar10_dataset()

    # train our model
    model = nn_fit(trainloader, 0.001, 2)

    # test the model
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy: %d %%' % (100 * correct / total))

if __name__ == "__main__":
    main()
