import os.path
import pandas as pd
from model import factory


if __name__ == '__main__':
    cur_path = os.path.abspath(os.path.dirname(__file__))
    data_path = os.path.join(cur_path, "../data/frame_data.txt")
    frame_data_init = pd.read_csv(data_path, sep='\t', index_col=0)
    forward_returns = frame_data_init.pop('FRET')

    srr_mis_ols = factory.RegressionRunFactory.mis_ols_of(frame_data_init, forward_returns)
    srr_mis_wls = factory.RegressionRunFactory.mis_wls_of(frame_data_init, forward_returns)
    srr_ms_ols = factory.RegressionRunFactory.ms_ols_of(frame_data_init, forward_returns)
    srr_ms_wls = factory.RegressionRunFactory.ms_wls_of(frame_data_init, forward_returns)

    print(srr_mis_ols.model_string, srr_mis_ols.factor_returns())
    print(srr_mis_wls.model_string, srr_mis_wls.factor_returns())
    print(srr_ms_ols.model_string, srr_ms_ols.factor_returns())
    print(srr_ms_wls.model_string, srr_ms_wls.factor_returns())
