import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def main():

    # import the loan data
    df = pd.read_csv('../data/iris.csv')

    # split the data into training and test sets
    train, test = train_test_split(df, test_size=0.3)

    # define our Decision Tree model
    clf = DecisionTreeClassifier()
    
    # "fit" our model
    clf.fit(train[['f1','f2','f3','f4']], train['species'])

    # calculate our prediction on the test set
    predictions = clf.predict(test[['f1','f2','f3','f4']])

    # calculate our accuracy
    acc = accuracy_score(test['species'], predictions)
    print('Accuracy: ', acc)

if __name__ == "__main__":
    main()
