# import libraries
import os
import sys
import warnings
import lightgbm
import numpy as np
import pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LassoLarsIC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeRegressor


class EvaluateMLModels:

    data = None
    n_jobs = None
    models = None
    scoring = None
    attr = None

    def __init__(self, data, attr, n_jobs=1):

        # requested mandatory parameters
        self.attr = attr

        self.data = data
        self.n_jobs = n_jobs

        # prepare models
        self.models = [
            ('LR', LinearRegression(n_jobs=self.n_jobs)),
            ('LassoLARS', LassoLarsIC()),
            ('Ridge', Ridge()),
            ('ElasticNet', ElasticNetCV(n_jobs=self.n_jobs)),
            ('LightGBM', lightgbm.sklearn.LGBMRegressor(n_jobs=self.n_jobs)),
            ('CART', DecisionTreeRegressor()),
            ('RandomForest', RandomForestRegressor())
        ]

        # prepare scoring metrics
        self.scoring = ['r2', 'neg_mean_squared_error']

    def train_single_var(self, yvar):

        if not sys.warnoptions:
            warnings.simplefilter("ignore")
            os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses

        # subset data
        comb_feat = self.attr + [yvar]
        sub_data = self.data[comb_feat]

        # setup X for training
        Xtr = sub_data.iloc[:, 0:len(self.attr)]

        # setup Y for training
        Ytr = sub_data.iloc[:, len(self.attr)]

        # setup 10-fold CV
        kf = KFold(n_splits=10, shuffle=True)

        # scores
        scores = []
        for name, model in self.models:

            # get cross validation scores
            sc = cross_validate(estimator=model, X=Xtr, y=Ytr, scoring=self.scoring, cv=kf, n_jobs=self.n_jobs)
            scores.append([name, np.mean(sc['test_r2']), np.mean(sc['test_neg_mean_squared_error'])])

        return pandas.DataFrame(scores, columns=['Name', 'R2', 'NegRMSE'])
