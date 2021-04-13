import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoLarsIC
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Ridge
import lightgbm
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import argparse
import os
import joblib
import time


class TrainSelectedModel:

    X = None
    Y = None
    model = None

    def __init__(self, X, Y, n_jobs=1, model_name='RandomForest'):

        self.X = X
        self.Y = Y

        # select a model according to 'model_name'
        if model_name == 'LR':
            self.model = LinearRegression(n_jobs=n_jobs)
        elif model_name == 'Ridge':
            self.model = Ridge()
        elif model_name == 'LassoLARS':
            self.model = LassoLarsIC()
        elif model_name == 'ElasticNet':
            self.model = ElasticNetCV(n_jobs=n_jobs)
        elif model_name == 'LightGBM':
            self.model = lightgbm.sklearn.LGBMRegressor(n_jobs=n_jobs)
        elif model_name == 'CART':
            self.model = DecisionTreeRegressor()
        else:
            self.model = RandomForestRegressor()

        # fit the model
        self.model.fit(self.X, self.Y)

    def GetModel(self):
        return self.model


def TrainMARM():

    # setup argument list
    arguments = setup_arguments()
    args = arguments.parse_args()

    # check if the path_to_data is valid
    if not os.path.isfile(args.path_to_data):
        print("Invalid path to cleaned data! Exiting..")
        raise SystemExit

    # check if path_to_marm_res is valid
    if not os.path.isdir(args.path_to_marm_res):
        print("Invalid path to MARM intermediate results! Exiting..")
        raise SystemExit

    # create output directory if it does not exist
    if not os.path.isdir(args.output_dir):
        print("Creating " + args.output_dir)
        os.makedirs(args.output_dir)

    # process indep_vars
    indep_vars = args.indep_vars.split(",")

    # load data
    data = joblib.load(args.path_to_data)

    # load results
    select_result_name = args.path_to_marm_res + "/" + args.sys_name + "_" + args.dep_vars + "_" + \
                         args.sel_model + "_MARM_report.csv"
    scores = pd.read_csv(select_result_name)

    # extract ids
    all_accs = scores['Account'].values.flatten()
    accids = []
    for k in range(args.num_acc):
        accids.append(all_accs[k])

    # extract data
    data = data[data['Account'].isin(accids)]
    X = data[indep_vars]
    Y = data[args.dep_vars].values.ravel()

    # begin building MARMs
    print("Building selected MARM.")
    start_time = time.time()

    smarm = TrainSelectedModel(X=X, Y=Y, model_name=args.sel_model)

    model = smarm.GetModel()

    end_time = time.time()
    print("MARM built in " + str(end_time - start_time) + " seconds.")

    # save model
    joblib.dump(model, args.output_dir + "/" + args.system_name + "_" + args.dep_vars + "_" + args.sel_model +
                "_MARM.pkl")


def setup_arguments():

    arguments = argparse.ArgumentParser()
    arguments.add_argument('-path_to_data', help='string; Full path to cleaned data obtained using PreProcess.',
                           required=True, type=str)
    arguments.add_argument('-path_to_marm_res', help='string; Full path to the directory containing intermediate '
                                                     'results produced by BuildMixedAccountModels.',
                           required=True, type=str)
    arguments.add_argument('-num_acc', help='int; Number of accounts to choose based on the results produced by'
                                            'BuildMixedAccountModels', required=True, type=int)
    arguments.add_argument('-output_dir', help='string; Path to output directory to save results. If such a directory '
                                               'does not exist, it will be created.', required=True, type=str)
    arguments.add_argument('-sys_name', help='string; Name of the HPC System.', required=True, type=str)
    arguments.add_argument('-dep_vars', help='string; Name of the response variable to regress '
                                             '(options include: CPUTimeRAW or MaxRSS).', required=True, type=str)
    arguments.add_argument('-indep_vars', help='string; A comma (,) separated name of factors to be used in '
                                               'building the regression models '
                                               '(default: Timelimit,ReqCPUS,ReqMem,ReqNodes,QOS).'
                           , required=False, default="Timelimit,ReqCPUS,ReqMem,ReqNodes,QOS", type=str)
    arguments.add_argument('-sel_model', help='string; Name of the selected ML model to be used to build the final '
                                              'MARM. Valid options include: LR, Ridge, LassoLARS, ElasticNet, '
                                              'RandomForest, CART and LightGBM (default: RandomForest).',
                           required=False, type=str, default="RandomForest")
    return arguments


TrainMARM()