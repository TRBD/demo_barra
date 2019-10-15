

class RegressionParams(object):
    """
    Column data configuration for regression endogenous and exogenous inputs

    Parameters for determination:
        -   model_string:
            Whether Market+Industry+Style or Market+Style, OLS, WLS, etc
            -   MIS_OLS: Market+Industry+Style, OLS
            -   MS_WLS: Market+Industry+Style, WLS
        -   exposure_col: list<column name> with categorical exposure codes
        -   numeric_cols: list<column names> with numeric exposure values
        -   ret_var: column with regressand


    """
    def __init__(self):
        self.model_string = None
        self.exposure_col = None
        self.numeric_cols = None
        self.ret_var = None
        self.weight_col = None

    @property
    def combined_exposure_vars(self):
        return self.exposure_col + self.numeric_cols

    def factor_category_names(self, frame_data):
        return sorted(
            frame_data[self.exposure_col[0]].drop_duplicates().tolist()
        )

    def factor_numeric_names(self, frame_input):
        return self.numeric_cols

    def factor_names(self, frame_input):
        return self.factor_category_names(frame_input) + \
            self.factor_numeric_names(frame_input)

    def __hash__(self):
        h0 = hash(self.model_string)
        if self.exposure_col is not None:
            for exposure_col in self.exposure_col:
                h0 += hash(exposure_col)
        if self.numeric_cols is not None:
            for num_col in self.numeric_cols:
                h0 += hash(num_col)
        h0 += hash(self.ret_var)
        return h0

    def __eq__(self, other):
        if self.__class__.__name__ != other.__class__.__name__:
            return False
        is_equal = self.model_string == other.model_string
        if not is_equal:
            return False
        if self.exposure_col is not None:
            if other.exposure_cols is None:
                return False
            if not set(self.exposure_col) == set(other.exposure_cols):
                return False
        if self.numeric_cols is not None:
            if other.num_cols is None:
                return False
            if not set(self.numeric_cols) == set(other.num_cols):
                return False
        return self.ret_var == other.ret_var

