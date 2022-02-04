import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from feature_engine.imputation import DropMissingData


# Dropping duplicate observations in a dataframe
class DropDuplicateData(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        # To accomodate sklearn pipeline
        return self
    
    def transform(self, X):
        X = X.copy()
        X = X[~X.duplicated()].reset_index(drop=True)
        
        return X

class Mapper(BaseEstimator, TransformerMixin):

    def __init__(self, variables, mappings):

        if not isinstance(variables, list):
            raise ValueError('variables should be a list')

        self.variables = variables
        self.mappings = mappings

    def fit(self, X, y=None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature].replace(self.mappings, inplace=True)

        return X

    
# Creating Training, Testing and Validation splits
def trainTestValid_split(X, y, trainsize, testsize, randomstate=0):
    trainX, test_tmpX, trainY, test_tmpY = train_test_split(X, y,
                                                            train_size=trainsize,
                                                            random_state=randomstate,
                                                            stratify = y)
    
    testX, valX, testY, valY = train_test_split(test_tmpX, test_tmpY,
                                                train_size=testsize,
                                                random_state=randomstate,
                                                stratify = test_tmpY)
    
    return(trainX, testX, valX, trainY, testY, valY)