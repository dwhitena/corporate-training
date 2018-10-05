import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def main():

    # import the loan data
    df = pd.read_csv('../data/iris.csv')

    # split the data into training and test sets
    train, test = train_test_split(df, test_size=0.3)

    # define and "fit" our kNN model
    nbrs = KNeighborsClassifier(n_neighbors=3)
    
    # "fit" our model
    nbrs.fit(train[['f1','f2','f3','f4']], train['species'])

    # calculate our prediction on the test set
    predictions = nbrs.predict(test[['f1','f2','f3','f4']])

    # calculate our accuracy
    acc = accuracy_score(test['species'], predictions)
    print('Accuracy: ', acc)

if __name__ == "__main__":
    main()
