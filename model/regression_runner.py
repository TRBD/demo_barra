import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import patsy

from regression_result import SnapRegressionResult
import constants


class RegressionRunBase(object):
    """
    Base class for various regression implementations

    model_function: An implementation of static class ModelFunction
    io_data_handler: An implementation of static class IODataHandler

    Standardized methods:
        run_regression(self, regression_dt, model_data, fwd_rets, params)
    """
    MODEL_STRING = None

    def __init__(self, model_function, io_data_handler, clean_function=None):
        self.model_function = model_function
        self.io_data_handler = io_data_handler
        if clean_function is None:
            self.clean_function = DataCleanFunction
        else:
            self.clean_function = clean_function

    def run_regression(self, model_data_frame, fwd_rets, params):
        # clean input data frame
        frame_scored_clean = self.clean_function.clean_input(model_data_frame, fwd_rets, params)
        # beta matrices
        beta_star = self.io_data_handler.model_beta_star(params, frame_scored_clean)
        beta_style = self.io_data_handler.model_beta_style(params, frame_scored_clean)
        k_category = beta_star.shape[1]

        r_matrix = self.io_data_handler.restriction_matrix(params, frame_scored_clean, beta_star, beta_style)

        formula = self.io_data_handler.formula(params, frame_scored_clean, beta_star, beta_style)

        beta_formula = self.io_data_handler.beta_formula(params, frame_scored_clean, beta_star, beta_style, r_matrix)

        # model fit
        m = self.model_function.instance_model(formula, beta_formula, frame_scored_clean)
        f = m.fit()

        # output formatting
        beta_combined = beta_star.join(beta_style)
        beta_combined.columns = map(lambda c: c.replace(params.exposure_col[0], '').
                                    replace('[', '').
                                    replace(']', ''),
                                    beta_combined.columns)

        # output population
        srr = SnapRegressionResult()
        srr.factor_category_names = self.io_data_handler.factor_category_names(params, frame_scored_clean)
        srr.regression_params = params
        srr.factor_numeric_names = self.io_data_handler.factor_numeric_names(params, frame_scored_clean)
        srr.factor_names = params.factor_names(frame_scored_clean)
        srr.n_assets = self.io_data_handler.n_assets(frame_scored_clean)

        srr.params = self.io_data_handler.output_regression_params(f, self.io_data_handler.output_param_names(
            params, frame_scored_clean, beta_star, beta_style, r_matrix))
        srr.rsq = f.rsquared
        srr.beta_matrix = beta_combined
        srr.residuals = f.resid
        srr.restriction_matrix = r_matrix
        srr.k_category = k_category
        srr.raw_weights = frame_scored_clean[constants.MODEL_MC_WEIGHT_COL]
        self.set_model_string(srr)
        return srr

    def set_model_string(self, regression_result):
        regression_result.model_string = self.MODEL_STRING


class DataCleanFunction(object):

    @classmethod
    def add_returns(cls, frame_clean, fwd_rets, params):
        frame_clean[params.ret_var] = fwd_rets.ix[frame_clean.index]
        return frame_clean

    @classmethod
    def process_frame_input(cls, model_data_frame, fwd_rets, params):
        return model_data_frame

    @classmethod
    def process_returns_input(cls, model_data_frame, fwd_rets, params):
        return fwd_rets

    @classmethod
    def clean_input(cls, model_data_frame, fwd_rets, params):
        model_data_frame_clean = model_data_frame.copy()
        # Clean up input values
        model_data_frame_clean = cls.process_frame_input(model_data_frame_clean, fwd_rets, params)
        # Clean up returns values
        fwd_rets_clean = cls.process_returns_input(model_data_frame_clean, fwd_rets, params)
        # add forward returns
        model_data_frame_clean = cls.add_returns(model_data_frame_clean, fwd_rets_clean, params)
        return model_data_frame_clean


class ModelFunction(object):
    """
    create_model function differentiates between OLS and WLS implementations
    """
    @classmethod
    def instance_model(cls, regression_formula, beta_formula, frame_input):
        raise NotImplementedError


class IODataHandler(object):
    """
    Static class for the implementation of various Regression types.

    Standardized methods:
        formula
        n_assets
        output_regression_params
        calc_beta_means
        calc_beta_stds
    Required Override methods:
        clean_input(self, model_data, fwd_rets, params)
        factor_category_names(self, params, frame_input)
        factor_numeric_names(self, params, frame_input)
        model_beta_star(self, params, frame_input)
        model_beta_style(self, params, frame_input)
        output_param_names(self, params, frame_input, beta_star, beta_style, restriction_matrix)
        restriction_matrix(self, params, frame_input, beta_star, beta_style)
        create_model(self, params, frame_input, beta_star, beta_style, restriction_matrix)
    """
    @classmethod
    def formula(cls, params, frame_input, beta_star, beta_style):
        """Returns the formula used by statsmodels formula api factory functions
        """
        return params.ret_var + '~' + 'beta_formula+' + '+'.join(params.numeric_cols) + '-1'

    @classmethod
    def n_assets(cls, frame_input):
        return frame_input.shape[0]

    @classmethod
    def output_regression_params(cls, stats_fit, params_names):
        """convenience for pulling the regression fit betas from the sm fit object
        """
        return pd.Series(stats_fit.params.values, index=params_names)

    @classmethod
    def factor_category_names(cls, params, frame_input):
        """returns correctly ordered category variable names
        """
        raise NotImplementedError

    @classmethod
    def factor_numeric_names(cls, params, frame_input):
        """returns correctly ordered numeric/style variable names
        """
        raise NotImplementedError

    @classmethod
    def model_beta_style(cls, params, frame_input):
        raise NotImplementedError

    @classmethod
    def model_beta_star(cls, params, frame_input):
        raise NotImplementedError

    @classmethod
    def restriction_matrix(cls, params, frame_input, beta_star, beta_style):
        raise NotImplementedError

    @classmethod
    def beta_formula(cls, params, frame_input, beta_star, beta_style, restriction_matrix):
        raise NotImplementedError

    @classmethod
    def output_param_names(cls, params, frame_input, beta_star, beta_style, restriction_matrix):
        raise NotImplementedError


class WLSModelFunction(ModelFunction):

    @classmethod
    def instance_model(cls, regression_formula, beta_formula, frame_input):
        wts = frame_input.MCAP / frame_input.MCAP.sum()
        wts = wts.fillna(0.)
        wts = wts.loc[frame_input.index]
        m = smf.wls(formula=regression_formula, data=frame_input, weights=wts)
        return m


class OLSModelFunction(ModelFunction):

    @classmethod
    def instance_model(cls, regression_formula, beta_formula, frame_input):
        m = smf.ols(formula=regression_formula, data=frame_input)
        return m


class StyleOnlyIODataHandler(IODataHandler):

    @classmethod
    def factor_category_names(cls, params, frame_input):
        return ['Market']

    @classmethod
    def factor_numeric_names(cls, params, frame_input):
        return params.numeric_cols

    @classmethod
    def beta_formula(cls, params, frame_input, beta_star, beta_style, restriction_matrix):
        return beta_star

    @classmethod
    def model_beta_style(cls, params, frame_input):
        fm_formula = params.ret_var + "~" + '+'.join(params.numeric_cols)
        fm_formula += ' - 1'
        beta = patsy.dmatrices(fm_formula, frame_input, return_type='dataframe')[1]
        beta_style = beta.ix[:, map(lambda k: beta.design_info.column_name_indexes[k], params.numeric_cols)]
        return beta_style

    @classmethod
    def model_beta_star(cls, params, frame_input):
        market_beta = pd.DataFrame(np.ones(frame_input.shape[0]), index=frame_input.index, columns=['Market'])
        beta_star = market_beta
        return beta_star

    @classmethod
    def restriction_matrix(cls, params, frame_input, beta_star, beta_style):
        return None

    @classmethod
    def output_param_names(cls, params, frame_input, beta_star, beta_style, restriction_matrix):
        return beta_star.join(beta_style).columns


class IndustryStyleIODataHandler(IODataHandler):
    @classmethod
    def factor_category_names(cls, params, frame_input):
        return ['Market'] + \
            params.factor_category_names(frame_input)

    @classmethod
    def factor_numeric_names(cls, params, frame_input):
        return params.numeric_cols

    @classmethod
    def output_param_names(cls, params, frame_input, beta_star, beta_style, restriction_matrix):
        beta_mod = cls.beta_formula(params, frame_input, beta_star, beta_style, restriction_matrix)
        beta_stms = beta_mod.join(beta_style)
        beta_stms.columns = map(lambda c: c.replace(params.exposure_col[0], '').
                                replace('[', '').
                                replace(']', ''),
                                beta_stms.columns)
        return beta_stms.columns

    @classmethod
    def beta_formula(cls, params, frame_input, beta_star, beta_style, restriction_matrix):
        return pd.DataFrame(
            np.dot(beta_star.iloc[:frame_input.shape[0], :], restriction_matrix),
            index=frame_input.index,
            columns=beta_star.columns[:-1])

    @classmethod
    def model_beta_style(cls, params, frame_input):
        fm_formula = params.ret_var + "~" + '+'.join(params.exposure_col + params.numeric_cols)
        fm_formula += ' - 1'
        beta = patsy.dmatrices(fm_formula, frame_input, return_type='dataframe')[1]
        beta_style = beta.ix[:, map(lambda k: beta.design_info.column_name_indexes[k], params.numeric_cols)]
        return beta_style

    @classmethod
    def model_beta_star(cls, params, frame_input):
        formula_expochar = params.ret_var + '~ ' + params.exposure_col[0] + ' - 1'
        beta_expochar = patsy.dmatrices(formula_expochar, frame_input, return_type='dataframe')[1]
        market_beta = pd.DataFrame(np.ones(beta_expochar.shape[0]), index=frame_input.index, columns=['Market'])
        beta_star = market_beta.join(beta_expochar)
        return beta_star

    @classmethod
    def restriction_matrix(cls, params, frame_input, beta_star, beta_style):
        k_category = beta_star.shape[1]
        sector_weights = (frame_input.groupby(params.exposure_col[0])[params.weight_col].sum() /
                          frame_input[params.weight_col].sum()).fillna(0.)
        restriction_weights = sector_weights.values[:-1] / sector_weights.values[-1]
        tmp = np.array([0] + (-restriction_weights).tolist())
        tmp2 = np.diag(np.ones(k_category-1))
        restriction_matrix = np.vstack([tmp2, tmp])
        return restriction_matrix


class IndustryStyleOLSRun(RegressionRunBase):
    """
    Class for an OLS Market + Industry + Style model where
        Industry returns are required to sum to zero

    """
    MODEL_STRING = 'MIS_OLS'

    def __init__(self, clean_function=None):
        super(IndustryStyleOLSRun, self).__init__(OLSModelFunction, IndustryStyleIODataHandler, clean_function)


class IndustryStyleWLSRun(RegressionRunBase):
    MODEL_STRING = 'MIS_WLS'

    def __init__(self, clean_function=None):
        super(IndustryStyleWLSRun, self).__init__(WLSModelFunction, IndustryStyleIODataHandler, clean_function)


class StyleOnlyOLSRun(RegressionRunBase):
    """
    Class for an OLS Market + Style model
    """
    MODEL_STRING = 'MS_OLS'

    def __init__(self, clean_function=None):
        super(StyleOnlyOLSRun, self).__init__(OLSModelFunction, StyleOnlyIODataHandler, clean_function)


class StyleOnlyWLSRun(RegressionRunBase):
    MODEL_STRING = 'MS_WLS'

    def __init__(self, clean_function=None):
        super(StyleOnlyWLSRun, self).__init__(WLSModelFunction, StyleOnlyIODataHandler, clean_function)


