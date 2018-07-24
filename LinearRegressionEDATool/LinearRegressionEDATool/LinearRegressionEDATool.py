import pandas as pd
from pltlearningcurve import plot_learning_curve
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from matplotlib.backends.backend_pdf import PdfPages


def linear_regression_eda_tool(features, target, requested_regressions=['Ridge']):

    """
    Generate the learning curve for multiple linear regressions and print them to a file
    Parameters
    ----------
    features : feature data to be used for your regression application

    target: target data to be used for your regression application

    requested_regressions: list of strings that contain the regressions you want to compare.
                           Supports all the types in class sklearn.linear_model.
                           See http://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model

    """
    
    regression_models = dict()
    for r in requested_regressions:
        get_regression_estimators(r, regression_models)

    i = 1
    output_file = PdfPages('LearningCurves.pdf')

    for m in regression_models:
       title = "Learning Curve for Regression model: " + m
       cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
       lc = plot_learning_curve(regression_models[m], title, features, target, ylim=(0.1, 0.9), cv=cv, n_jobs=4)
       scores = cross_val_score(regression_models[m], features, target,cv=cv)
       scores_string = "Accuracy of cross-validation scores: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
       lc.text(0,0,scores_string)
       lc.figure(i)
       output_file.savefig()
       i = i+1

    output_file.close()

def get_regression_estimators(r, regression_models):
    if r == 'ARDRegression':
        regression_models[r]=linear_model.ARDRegression()
    elif r == 'BayesianRidge':
        regression_models[r]=linear_model.BayesianRidge()
    elif r == 'ElasticNet':
        regression_models[r]=linear_model.ElasticNet()
    elif r == 'ElasticNetCV':
        regression_models[r]=linear_model.ElasticNetCV()
    elif r == 'HuberRegressor':
        regression_models[r]=linear_model.HuberRegressor()
    elif r == 'Lars':
        regression_models[r]=linear_model.Lars()
    elif r == 'LarsCV':
        regression_models[r]=linear_model.LarsCV()
    elif r == 'Lasso':
        regression_models[r]=linear_model.Lasso()
    elif r == 'LassoCV':
        regression_models[r]=linear_model.LassoCV()
    elif r == 'LassoLars':
        regression_models[r]=linear_model.LassoLars()
    elif r == 'LassoLarsCV':
        regression_models[r]=linear_model.LassoLarsCV()
    elif r == 'LassoLarsIC':
        regression_models[r]=linear_model.LassoLarsIC()
    elif r == 'LinearRegression':
        regression_models[r]=linear_model.LinearRegression()
    elif r == 'LogisticRegression':
        regression_models[r]=linear_model.LogisticRegression()
    elif r == 'LogisticRegressionCV':
        regression_models[r]=linear_model.LogisticRegressionCV()
    elif r == 'MultiTaskElasticNet':
        regression_models[r]=linear_model.MultiTaskElasticNet()
    elif r == 'MultiTaskElasticNetCV':
        regression_models[r]=linear_model.MultiTaskElasticNetCV()
    elif r == 'MultiTaskLasso':
        regression_models[r]=linear_model.MultiTaskLasso()
    elif r == 'MultiTaskLassoCV':
        regression_models[r]=linear_model.MultiTaskLassoCV()
    elif r == 'OrthogonalMatchingPursuit':
        regression_models[r]=linear_model.OrthogonalMatchingPursuit()
    elif r == 'OrthogonalMatchingPursuitCV':
        regression_models[r]=linear_model.OrthogonalMatchingPursuitCV()
    elif r == 'PassiveAggressiveClassifier':
        regression_models[r]=linear_model.PassiveAggressiveClassifier()
    elif r == 'PassiveAggressiveRegressor':
        regression_models[r]=linear_model.PassiveAggressiveRegressor()
    elif r == 'Perceptron':
        regression_models[r]=linear_model.Perceptron()
    elif r == 'RANSACRegressor':
        regression_models[r]=linear_model.RANSACRegressor()
    elif r == 'Ridge':
        regression_models[r]=linear_model.Ridge()
    elif r == 'RidgeClassifier':
        regression_models[r]=linear_model.RidgeClassifier()
    elif r == 'RidgeClassifierCV':
        regression_models[r]=linear_model.RidgeClassifierCV()
    elif r == 'RidgeCV':
        regression_models[r]=linear_model.RidgeCV()
    elif r == 'SGDClassifier':
        regression_models[r]=linear_model.SGDClassifier()
    elif r == 'SGDRegressor':
        regression_models[r]=linear_model.SGDRegressor()
    elif r == 'TheilSenRegressor':
        regression_models[r]=linear_model.TheilSenRegressor()
    else:
        print(r + " is an unsupported regression type. Check if you have misspelled the name.")


