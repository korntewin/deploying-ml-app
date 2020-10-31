from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer


class LogTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        self.variables = variables
        self.transformer = PowerTransformer()

    def fit(self, X, y=None):
        self.transformer.fit(X[self.variables])
        return self

    def transform(self, X):
        df = X.copy()
        df[self.variables] = self.transformer.transform(df[self.variables])

        return df