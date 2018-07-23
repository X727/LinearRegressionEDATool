import pandas as pd
from pltlearningcurve import plot_learning_curve
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import ShuffleSplit

def linear_regression_eda_tool(features, target, requested_regressions=['Linear']):
    
    regression_models = dict()
    for r in requested_regressions:
        if r == 'Linear':
            from sklearn.linear_model import LinearRegression
            regression_models[r]=LinearRegression()
        elif r == 'LASSO':
            from sklearn.linear_model import Lasso
            regression_models[r]=Lasso()
        else:
            print(r + " is an unsupported regression type")

    i = 1
    for m in regression_models:
       title = "Learning Curve for Regression model: " + m
       cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
       lc = plot_learning_curve(regression_models[m], title, features, target, ylim=(0.1, 0.9), cv=cv, n_jobs=4)
       lc.figure(i)
       i = i+1

    lc.show()


