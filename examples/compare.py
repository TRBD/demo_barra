import os.path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from model import factory
from model import regression_runner


if __name__ == '__main__':
    cur_path = os.path.abspath(os.path.dirname(__file__))
    data_path = os.path.join(cur_path, "../data/frame_data.txt")
    frame_data_init = pd.read_csv(data_path, sep='\t', index_col=0)
    forward_returns = frame_data_init.pop('FRET')
    srr_mis_ols = factory.RegressionRunFactory.mis_ols_of(frame_data_init, forward_returns)
    model_returns = srr_mis_ols.factor_returns()
    params = factory.RegressionRunFactory.generate_params()
    frame_scored_clean = regression_runner.DataCleanFunction.clean_input(frame_data_init, forward_returns, params)

    beta_star = regression_runner.IndustryStyleIODataHandler.model_beta_star(params, frame_scored_clean)
    beta_style = regression_runner.IndustryStyleIODataHandler.model_beta_style(params, frame_scored_clean)
    beta_raw = beta_star.join(beta_style)

    X = beta_raw
    sector_weights = (frame_scored_clean.groupby(params.exposure_col[0])[params.weight_col].sum()
                          / frame_scored_clean[params.weight_col].sum()).fillna(0.)

    R = np.array([0] + sector_weights.tolist() + [0,0,0,0,0,0])
    R = np.atleast_2d(R)
    r = np.array([[0]])
    raw_b = sm.OLS(frame_scored_clean.FRET, beta_raw).fit().params
    raw_b = np.atleast_2d(raw_b).T

    exact = raw_b - np.dot(
        np.dot(
            np.linalg.inv(
                np.dot(X.T, X)),
            R.T), np.dot(
            np.linalg.inv(
                np.dot(
                    np.dot(
                        R, np.linalg.inv(np.dot(X.T, X))), R.T)),
            np.dot(R, raw_b) - r))

    exact_returns = pd.Series(exact.flatten(), index=model_returns.index)

    print(srr_mis_ols.model_string, model_returns)
    print('Long form', exact_returns)
    print('Max Abs Diff', (model_returns - exact_returns).abs().max())

