import string
import regex as re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin


class textcleanup(BaseEstimator, TransformerMixin):
    
    def __init__(self, variable, reference_variable=None):
        
        if not isinstance(variable, str):
            raise ValueError('variable should be a string')
        
        self.variable = variable
        if reference_variable:
            self.reference_variable = reference_variable
        else:
            self.reference_variable = variable
    
    
    def fit(self, X, y=None):
        
        return self
    
    
    def transform(self, X):
        
        X = X.copy()
        X[self.reference_variable] = X[self.variable].apply(
            lambda x: re.sub(r' +', ' ', # Replace extra space by a single space
                             re.sub(r'x{2,}', ' ', # Replace masked values by single space
                                    re.sub(r'[^a-z]', ' ', # Replace all values other than alphabets with single space
                                           x.lower())))) # Convert all alphabets to lower case
        X[self.reference_variable] = X[self.reference_variable].str.strip() # Remove leading and trailing spaces
        
        return X

    
class texttokenize(BaseEstimator, TransformerMixin):
    
    def __init__(self, variable, reference_variable=None):
        
        if not isinstance(variable, str):
            raise ValueError('variable should be a string')
        
        self.variable = variable
        if reference_variable:
            self.reference_variable = reference_variable
        else:
            self.reference_variable = variable
    
    
    def fit(self, X, y=None):
        
        return self
    
    
    def transform(self, X):
        
        X = X.copy()
        X[self.reference_variable] = X[self.variable].apply(word_tokenize)
        
        return X


class textstopwordremove(BaseEstimator, TransformerMixin):
    
    def __init__(self, variable, reference_variable=None):
        
        if not isinstance(variable, str):
            raise ValueError('variable should be a string')
        
        self.variable = variable
        if reference_variable:
            self.reference_variable = reference_variable
        else:
            self.reference_variable = variable
    
    
    def fit(self, X, y=None):
        
        return self
    
    
    def transform(self, X):
        
        X = X.copy()
        Stopwords = stopwords.words('English')
        Stopwords += [i for i in string.ascii_lowercase]
        X[self.reference_variable] = X[self.variable].apply(lambda x: [i for i in x if i not in Stopwords])
        
        return X

   
class textlemmatize(BaseEstimator, TransformerMixin):
    
    Stopwords = stopwords.words('English')
    Stopwords += [i for i in string.ascii_lowercase]
    
    def __init__(self, variable, reference_variable=None):
        
        if not isinstance(variable, str):
            raise ValueError('variable should be a string')
        
        self.variable = variable
        if reference_variable:
            self.reference_variable = reference_variable
        else:
            self.reference_variable = variable
    
    
    def fit(self, X, y=None):
        
        return self
    
    
    def transform(self, X):
        
        X = X.copy()
        lemmatize = WordNetLemmatizer()
        X[self.reference_variable] = X[self.variable].apply(lambda x: [lemmatize.lemmatize(i) for i in x])
        
        return X


class textstemmer(BaseEstimator, TransformerMixin):
    
    Stopwords = stopwords.words('English')
    Stopwords += [i for i in string.ascii_lowercase]
    
    def __init__(self, variable, reference_variable=None):
        
        if not isinstance(variable, str):
            raise ValueError('variable should be a string')
        
        self.variable = variable
        if reference_variable:
            self.reference_variable = reference_variable
        else:
            self.reference_variable = variable
    
    
    def fit(self, X, y=None):
        
        return self
    
    
    def transform(self, X):
        
        X = X.copy()
        stemmer = PorterStemmer()
        X[self.reference_variable] = X[self.variable].apply(lambda x: ' '.join([stemmer.stem(i) for i in x]))
        X[self.reference_variable] = X[self.reference_variable].str.strip() # Remove leading and trailing spaces
        
        return X
