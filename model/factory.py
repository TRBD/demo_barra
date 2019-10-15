import constants
from params import RegressionParams
from regression_runner import IndustryStyleOLSRun, IndustryStyleWLSRun, StyleOnlyOLSRun, StyleOnlyWLSRun


class RegressionRunFactory(object):

    @classmethod
    def mis_ols_of(cls, model_data_frame, fwd_rets, clean_function=None):
        regression_params = cls.generate_params()
        runner = IndustryStyleOLSRun(clean_function)
        return cls.run_regression(runner,
                                  model_data_frame,
                                  fwd_rets,
                                  regression_params)

    @classmethod
    def mis_wls_of(cls, model_data_frame, fwd_rets, clean_function=None):
        regression_params = cls.generate_params()
        runner = IndustryStyleWLSRun(clean_function)
        return cls.run_regression(runner,
                                  model_data_frame,
                                  fwd_rets,
                                  regression_params)

    @classmethod
    def ms_ols_of(cls, model_data_frame, fwd_rets, clean_function=None):
        regression_params = cls.generate_params()
        runner = StyleOnlyOLSRun(clean_function)
        return cls.run_regression(runner,
                                  model_data_frame,
                                  fwd_rets,
                                  regression_params)

    @classmethod
    def ms_wls_of(cls, model_data_frame, fwd_rets, clean_function=None):
        regression_params = cls.generate_params()
        runner = StyleOnlyWLSRun(clean_function)
        return cls.run_regression(runner,
                                  model_data_frame,
                                  fwd_rets,
                                  regression_params)

    @classmethod
    def generate_params(cls):
        regression_params = RegressionParams()
        regression_params.numeric_cols = constants.MODEL_NUMERIC_COLS
        regression_params.exposure_col = constants.MODEL_INDUSTRY_COLS
        regression_params.ret_var = constants.MODEL_ENDOGENOUS
        regression_params.weight_col = constants.MODEL_MC_WEIGHT_COL
        return regression_params

    @classmethod
    def run_regression(cls, runner, model_data_frame, fwd_rets, regression_params):
        result = runner.run_regression(model_data_frame,
                                       fwd_rets,
                                       regression_params)
        return result
