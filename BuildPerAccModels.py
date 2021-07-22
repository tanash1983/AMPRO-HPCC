from EvaluateMLModels import EvaluateMLModels
# import libraries
import joblib
import numpy as np
import pandas
from sklearn.preprocessing import MinMaxScaler
import collections
import argparse
import os
import time
import matplotlib.pyplot as plt


class BuildPerAccModels:
    path_to_data = None
    data = None
    output_dir = None
    sys_name = None
    min_num_jobs = None
    indep_vars = None
    dep_var = None

    def __init__(self,
                 path_to_data,
                 output_dir,
                 sys_name,
                 dep_var,
                 min_num_jobs,
                 indep_vars):

        # assign class variables
        self.path_to_data = path_to_data
        self.output_dir = output_dir
        self.dep_var = dep_var
        self.min_num_jobs = min_num_jobs
        self.indep_vars = indep_vars
        self.sys_name = sys_name

        # load data
        self.data = joblib.load(path_to_data)

        # filter accounts with num of jobs less than min_num_jobs
        user_num = collections.Counter(map(str, self.data[['Account']].values.flatten()))
        ucount = list(user_num.values())
        unames = list(user_num.keys())

        unames = [unames[i] for i in range(len(ucount)) if ucount[i] > self.min_num_jobs]
        ucnt = [ucount[i] for i in range(len(ucount)) if ucount[i] > self.min_num_jobs]

        # subset data
        self.data = self.data[self.data['Account'].isin(list(map(int, unames)))]

        # create pandas dataframe
        userinfo = pandas.DataFrame({'AccID': unames, 'JobCount': ucnt})

        # save user info
        userinfo.to_csv(self.output_dir + "/" + self.sys_name + "_accinfo.csv")

    def Getranking(self, ranking, names, order=1):
        minmax = MinMaxScaler()
        ranks = minmax.fit_transform(order * np.array([ranking]).T).T[0]
        ranks = map(lambda x: round(x, 2), ranks)
        return dict(zip(names, ranks))

    def EvalModel(self, train_data, account_name):
        # evaluate all ML models
        mlmodels = EvaluateMLModels(data=train_data, attr=self.indep_vars)

        # accumulate results
        res = mlmodels.train_single_var(self.dep_var)

        # write to csv
        res.to_csv(path_or_buf=self.output_dir + "/" + self.sys_name + "_" + account_name
                               + "_" + self.dep_var + "_Results10FoldCV.csv")

        # return the best model name
        # treat r2 score
        var_r2 = res[['R2']].values.flatten()
        var_r2 = [0 if i < 0 else i for i in var_r2]

        ranks_var_res_r2 = self.Getranking(var_r2, res[['Name']].values.flatten(), order=1)
        ranks_var_res_nrmse = self.Getranking(res[['NegRMSE']].values.flatten(), res[['Name']].values.flatten(),
                                              order=1)
        ranks_var = pandas.DataFrame({'Name': list(ranks_var_res_r2.keys()),
                                      'Ranking': [(x1 + x2) / 2 for x1, x2 in zip(ranks_var_res_r2.values(),
                                                                                  ranks_var_res_nrmse.values())]})
        ranks_var = ranks_var.sort_values('Ranking', ascending=False)

        model_name = ranks_var[['Name']].values[0][0]

        return model_name

    def PerAccountModels(self):

        # accumulate the name of top models for each account
        top_model = []
        accounts = []

        # read accinfo
        accinfo = pandas.read_csv(self.output_dir + "/" + self.sys_name + "_accinfo.csv")
        accids = accinfo[['AccID']].values.flatten()
        for i in accids:
            # subset data according to account
            user_data = self.data[self.data['Account'].isin([i])]

            # get account name
            account_name = "Account" + "_" + str(i)
            accounts.append(account_name)

            print("Acount ID: ", account_name, "NSIZE: ", user_data.shape[0])

            # get the best model for dep_var given indep_var
            DT0 = self.data.copy()
            model_name = self.EvalModel(DT0, account_name)
            top_model.append(model_name)

        # prepare results
        peraccres = pandas.DataFrame({'Accounts': accounts,
                                      'Top Models': top_model})
        peraccres.to_csv(path_or_buf=self.output_dir + "/" + self.sys_name + "_" + self.dep_var
                                     + "_TopModelPerAccount.csv")


def fixr2(x):

    if x < 0:
        return 0
    else:
        return x


def GenerateReport(output_dir, yvar, sys_name):

    # read accinfo
    accinfo = pandas.read_csv(output_dir + "/" + sys_name + "_accinfo.csv")
    accids = accinfo[['AccID']].values

    # accumulate r2 and rmse over accids
    for i in accids:

        # get account name
        account_name = "Account" + "_" + str(i[0])

        # get file
        filename = output_dir + "/" + sys_name + "_" + account_name + "_" + yvar + "_Results10FoldCV.csv"

        # construct or join to dataframes of r2 and neg rmse
        df = pandas.read_csv(filename)
        tempr2 = df[['Name', 'R2']]
        temprmse = df[['Name', 'NegRMSE']]
        if i == accids[0][0]:
            dfr2 = tempr2
            dfr2 = dfr2.set_index('Name')
            dfrmse = temprmse
            dfrmse = dfrmse.set_index('Name')
        else:
            dfr2 = dfr2.join(tempr2.set_index('Name'), rsuffix="_" + account_name)
            dfrmse = dfrmse.join(temprmse.set_index('Name'), rsuffix="_" + account_name)

    # change names of first columns
    dfr2 = dfr2.rename(columns={'R2': 'R2_Account_' + str(accids[0][0])}).T
    dfrmse = dfrmse.rename(columns={'NegRMSE': 'NegRMSE_Account_' + str(accids[0][0])}).T

    # take care of negative r2
    dfr2 = dfr2.applymap(func=fixr2)

    # plot r2 scores
    plt.rc('xtick', labelsize=7)
    dfr2.boxplot()
    plt.grid(color='gray', linewidth=0.1)
    plt.title(sys_name + " " + yvar + " 10 Fold CV Report on R2")
    plt.xlabel("Methods")
    plt.ylabel("R2")
    plt.savefig(output_dir + "/" + sys_name + "_" + yvar + "_10FoldCV_R2_Report.pdf")
    plt.close()

    # plot negrmse scores
    plt.rc('xtick', labelsize=7)
    dfrmse.boxplot()
    plt.grid(color='gray', linewidth=0.1)
    plt.title(sys_name + " " + yvar + " 10 Fold CV Report on Negative RMSE")
    plt.xlabel("Methods")
    plt.ylabel("Negative RMSE")
    plt.savefig(output_dir + "/" + sys_name + "_" + yvar + "_10FoldCV_NegRMSE_Report.pdf")
    plt.close()

    # prepare reporting values for r2
    report_r2 = pandas.DataFrame({'Mean_R2': dfr2.mean(axis=0),
                                  'Median_R2': dfr2.median(axis=0),
                                  'LowerQ_R2': dfr2.quantile(q=0.25, axis=0),
                                  'UpperQ_R2': dfr2.quantile(q=0.75, axis=0),
                                  'Variance_R2': dfr2.var(axis=0)})
    report_r2.to_csv(output_dir + "/" + sys_name + "_" + yvar + "_10FoldCV_R2_Report.csv")

    # prepare reporting values for negrmse
    report_negrmse = pandas.DataFrame({'Mean_NegRMSE': dfrmse.mean(axis=0),
                                       'Median_NegRMSE': dfrmse.median(axis=0),
                                       'LowerQ_NegRMSE': dfrmse.quantile(q=0.25, axis=0),
                                       'UpperQ_NegRMSE': dfrmse.quantile(q=0.75, axis=0),
                                       'Variance_NegRMSE': dfrmse.var(axis=0)})
    report_negrmse.to_csv(output_dir + "/" + sys_name + "_" + yvar + "_10FoldCV_NegRMSE_Report.csv")


def BuildPerAccountModel():

    # setup argument list
    arguments = setup_arguments()
    args = arguments.parse_args()

    # check if the path_to_data is valid
    if not os.path.isfile(args.path_to_data):
        print("Invalid path to cleaned data! Exiting..")
        raise SystemExit

    # create output directory if it does not exist
    if not os.path.isdir(args.output_dir):
        print("Creating " + args.output_dir)
        os.makedirs(args.output_dir)

    # process indep_vars
    indep_vars = args.indep_vars.split(",")

    # begin building per account models
    print("Starting to build per-account models.")
    start_time = time.time()

    bpam = BuildPerAccModels(path_to_data=args.path_to_data, output_dir=args.output_dir, sys_name=args.sys_name,
                             dep_var=args.dep_vars, min_num_jobs=args.min_num_jobs, indep_vars=indep_vars)
    bpam.PerAccountModels()

    end_time = time.time()
    print("Per-account models built and evaluated in " + str(round(end_time - start_time, 2)) + " seconds.")

    # generate report
    GenerateReport(output_dir=args.output_dir, sys_name=args.sys_name, yvar=args.dep_vars)


def setup_arguments():
    arguments = argparse.ArgumentParser()
    arguments.add_argument('-path_to_data', help='string; Full path to cleaned data obtained using PreProcess.',
                           required=True, type=str)
    arguments.add_argument('-output_dir', help='string; Path to output directory to save results. If such a directory '
                                               'does not exist, it will be created.', required=True, type=str)
    arguments.add_argument('-sys_name', help='string; Name of the HPC System.', required=True, type=str)
    arguments.add_argument('-dep_vars', help='string; Name of the response variable to regress '
                                             '(options include: CPUTimeRAW or MaxRSS).', required=True, type=str)
    arguments.add_argument('-indep_vars', help='string; A semi-colon (,) separated name of factors to be used in '
                                               'building the regression models '
                                               '(default: Timelimit,ReqCPUS,ReqMem,ReqNodes,QOS).'
                           , required=False, default="Timelimit,ReqCPUS,ReqMem,ReqNodes,QOS", type=str)
    arguments.add_argument('-min_num_jobs', help='int; The minimum number of jobs under an account for it to be '
                                                 'considered for single and mixed account modeling (default: 1000).',
                           required=False, type=int, default=1000)
    return arguments


BuildPerAccountModel()
