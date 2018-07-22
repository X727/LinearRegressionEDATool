import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston

sns.set_style("whitegrid")

def main():
    boston_data = load_boston()
    boston_data_features = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
    boston_data_target = boston_data.target




if __name__ == '__main__':
    main()
