import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
import LinearRegressionEDATool

sns.set_style("whitegrid")

def main():
    boston_data = load_boston()
    boston_data_features = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
    #boston_data_features =  boston_data_features.values.reshape(-1,1)
    boston_data_target = boston_data.target
    LinearRegressionEDATool.LinearRegressionEDATool(boston_data_features,boston_data_target)




if __name__ == '__main__':
    main()
