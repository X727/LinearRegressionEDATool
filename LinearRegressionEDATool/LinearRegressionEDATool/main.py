import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from LinearRegressionEDATool import linear_regression_eda_tool

sns.set_style("whitegrid")

def main():
    boston_data = load_boston()
    boston_data_features = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
    boston_data_target = boston_data.target
    linear_regression_eda_tool(boston_data_features,boston_data_target, [ 'Ridge', 'BayesianRidge'])




if __name__ == '__main__':
    main()
