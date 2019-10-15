import numpy as np
import pandas as pd


class SnapRegressionResult(object):
    """
    Holds output from Statsmodels Fit object, required for further use
    Attributes:
        model_string: str
        params: pd.Series
        factor_names: list<str>
        factor_category_names: list<str>
        factor_numeric_names: list<str>
        beta_matrix: pd.DataFrame
        residuals: pd.Series
        rsq: float
        restriction_matrix: np.array or None
        k_category: int
        n_assets: int

    Store:
        x model_string: str
        x rsq float
        x k_category  int
        x n_assets    int
        x params  series
        x factor_category_names   list<str>
        x factor_numeric_names    list<str>
        x beta_matrix frame
        x residuals   series
        x restriction_matrix array
    """
    def __init__(self):
        self.params = None
        self.regression_params = None
        self.factor_names = None
        self.factor_category_names = None
        self.factor_numeric_names = None
        self.beta_matrix = None
        self.residuals = None
        self.raw_weights = None
        self.rsq = None
        self.restriction_matrix = None
        self.k_category = None
        self.n_assets = None

    @property
    def model_string(self):
        return self.regression_params.model_string

    @model_string.setter
    def model_string(self, model_string):
        self.regression_params.model_string = model_string

    def factor_returns_names(self):
        return self.factor_category_names + self.factor_numeric_names

    def factor_returns(self):
        if self.restriction_matrix is not None:
            fr0 = np.dot(self.restriction_matrix, self.params.values[:self.k_category-1])
            fac_rets = np.concatenate((fr0, self.params[self.k_category-1:].values))
            fac_rets = pd.Series(fac_rets,
                                 index=self.factor_returns_names())
        else:
            fac_rets = self.params.values
            fac_rets = pd.Series(fac_rets,
                                 index=self.factor_returns_names())
        return fac_rets

    def __hash__(self):
        h0 = hash(self.params)
        h0 += hash(self.factor_names)
        h0 += hash(self.factor_category_names)
        h0 += hash(self.factor_numeric_names)
        h0 += hash(self.beta_matrix)
        h0 += hash(self.residuals)
        h0 += hash(self.rsq)
        h0 += hash(self.restriction_matrix)
        h0 += hash(self.k_category)
        h0 += hash(self.n_assets)
        return h0

    def __eq__(self, other):
        if self.__class__.__name__ != other.__class__.__name__:
            return False
        is_equal = self.params == other.params and \
            self.factor_names == other.factor_names and \
            self.factor_category_names == other.factor_category_names and \
            self.factor_numeric_names == other.factor_numeric_names and \
            self.beta_matrix == other.beta_matrix and \
            self.residuals == other.residuals and \
            self.rsq == other.rsq and \
            self.restriction_matrix == other.restriction_matrix and \
            self.k_category == other.k_category and \
            self.n_assets == other.n_assets
        return is_equal

