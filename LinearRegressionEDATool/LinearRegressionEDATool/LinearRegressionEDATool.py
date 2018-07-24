import pandas as pd
from pltlearningcurve import plot_learning_curve
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import ShuffleSplit
from matplotlib.backends.backend_pdf import PdfPages

def linear_regression_eda_tool(features, target, requested_regressions=['Linear']):
    
    regression_models = dict()
    for r in requested_regressions:
        if r == 'Linear':
            from sklearn.linear_model import LinearRegression
            regression_models[r]=LinearRegression()
        elif r == 'Ridge':
            from sklearn.linear_model import Ridge
            regression_models[r]=Ridge()
        elif r == 'Lasso':
            from sklearn.linear_model import Lasso
            regression_models[r]=Lasso()
        elif r == 'MultiTaskLasso':
            from sklearn.linear_model import MultiTaskLasso
            regression_models[r]=MultiTaskLasso()
        elif r == 'ElasticNet':
            from sklearn.linear_model import ElasticNet
            regression_models[r]=ElasticNet()
        elif r == 'MultiTaskElasticNet':
            from sklearn.linear_model import MultiTaskElasticNet
            regression_models[r]=MultiTaskElasticNet()
        elif r == 'Lars':
            from sklearn.linear_model import Lars
            regression_models[r]=Lars()
        elif r == 'LassoLars':
            from sklearn.linear_model import LassoLars
            regression_models[r]=LassoLars()
        elif r == 'OrthogonalMatchingPursuit':
            from sklearn.linear_model import OrthogonalMatchingPursuit
            regression_models[r]=OrthogonalMatchingPursuit()
        elif r == 'BayesianRidge':
            from sklearn.linear_model import BayesianRidge
            regression_models[r]=BayesianRidge()
        elif r == 'ARDRegression':
            from sklearn.linear_model import ARDRegression
            regression_models[r]=ARDRegression()
        elif r == 'SGDRegressor':
            from sklearn.linear_model import SGDRegressor
            regression_models[r]=SGDRegressor()
        elif r == 'RANSACRegressor':
            from sklearn.linear_model import RANSACRegressor
            regression_models[r]=RANSACRegressor()
        elif r == 'TheilSenRegressor':
            from sklearn.linear_model import TheilSenRegressor
            regression_models[r]=TheilSenRegressor()
        elif r == 'HuberRegressor':
            from sklearn.linear_model import HuberRegressor
            regression_models[r]=HuberRegressor()
        else:
            print(r + " is an unsupported regression type. Check if you have misspelled the name.")

    i = 1
    output_file = PdfPages('LearningCurves.pdf')

    for m in regression_models:
       title = "Learning Curve for Regression model: " + m
       cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
       lc = plot_learning_curve(regression_models[m], title, features, target, ylim=(0.1, 0.9), cv=cv, n_jobs=4)
       lc.figure(i)
       output_file.savefig()
       i = i+1

    output_file.close()


