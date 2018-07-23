import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import ShuffleSplit
import plot_learning_curve

def LinearRegressionEDATool(features, target, requested_regressions=['Linear']):
    
    regression_models = []
    for r in requested_regressions:
        if r == 'Linear':
            from sklearn.linear_model import LinearRegression
            regression_models.append(LinearRegression())
        else:
            print(r + " is an unsupported regression type")

    for m in regression_models:
       title = "Learning Curve"
       cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
       lc = plot_learning_curve.plot_learning_curve(m, title, features, target, ylim=(0.1, 0.9), cv=cv, n_jobs=4)
       lc.show()


