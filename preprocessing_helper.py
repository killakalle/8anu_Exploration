import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    """Returns a DataFrame with only those columns that are specified in `attribute_names`"""
    
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        return X[self.attribute_names].values


class StringImputer(BaseEstimator, TransformerMixin):
      
    def __init__(self):
        """Impute missing values in a DataFrame with most frequent value in column.
        
        Feed me with a DataFrame - not just a pd.Series!"""
    
    def fit(self, X, y=None):
        self.fill = pd.Series([X[column].value_counts().index[0] for column in X],
                             index=X.columns)
        return self
    
    def transform(self, X, y=None):
        return X.fillna(self.fill)