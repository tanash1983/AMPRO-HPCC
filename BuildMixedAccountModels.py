# import libraries
import joblib
import numpy
import lightgbm
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import argparse
import collections
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LassoLarsIC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge


class BuildMixedAccountModels:
    path_to_data = None
    data = None
    methodnames = None
    accinfo = None
    accids = None
    indep_var = None
    dep_var = None
    score_collector = None
    models = None

    def __init__(self,
                 path_to_data,
                 methodnames,
                 min_num_jobs,
                 indep_var,
                 dep_var):

        # assign class variables
        self.path_to_data = path_to_data
        self.methodnames = methodnames
        self.indep_var = indep_var
        self.dep_var = dep_var

        # read data
        self.data = joblib.load(self.path_to_data)

        # filter accounts with num of jobs less than min_num_jobs
        user_num = collections.Counter(map(str, self.data[['Account']].values.flatten()))
        ucount = list(user_num.values())
        unames = list(user_num.keys())

        unames = [unames[i] for i in range(len(ucount)) if ucount[i] > min_num_jobs]
        ucnt = [ucount[i] for i in range(len(ucount)) if ucount[i] > min_num_jobs]

        # subset data
        self.data = self.data[self.data['Account'].isin(list(map(int, unames)))]

        # create pandas dataframe
        self.accinfo = pd.DataFrame({'AccID': unames, 'JobCount': ucnt})

        # subset data using accounts from accinfo
        self.accids = self.accinfo['AccID'].values.flatten()
        self.data = self.data[self.data['Account'].isin(self.accids)]
        self.data = self.data.reset_index()

        # prepare model container and score collector
        self.models = []
        self.score_collector = []
        for i in methodnames:

            # select a model according to i
            if i == 'LR':
                self.models.append(LinearRegression(n_jobs=-1))
            elif i == 'Ridge':
                self.models.append(Ridge())
            elif i == 'LassoLARS':
                self.models.append(LassoLarsIC())
            elif i == 'ElasticNet':
                self.models.append(ElasticNetCV(n_jobs=-1))
            elif i == 'LightGBM':
                self.models.append(lightgbm.sklearn.LGBMRegressor(n_jobs=-1))
            elif i == 'CART':
                self.models.append(DecisionTreeRegressor())
            elif i == 'RandomForest':
                self.models.append(RandomForestRegressor(n_jobs=-1))
            else:
                print("Invalid model name, exiting..")
                raise SystemExit

            self.score_collector.append(pd.DataFrame(0.0, index=range(int(numpy.round(0.8 * len(self.accids)))),
                                                     columns=['Account', 'Size', 'TrainR2', 'TestR2']))

    def Getranking(self, ranking, names, order=1):
        minmax = MinMaxScaler()
        ranks = minmax.fit_transform(order * numpy.array([ranking]).T).T[0]
        ranks = map(lambda x: round(x, 2), ranks)
        return dict(zip(names, ranks))

    def MARM(self):

        for m in range(len(self.models)):

            print("Building MARM on " + self.methodnames[m])

            # prepare data structures
            acc_so_far = []
            size_so_far = []
            best_tr_r2 = []
            best_te_r2 = []

            # we only do this for 80% of the users
            for j in range(self.score_collector[m].shape[0]):

                print("Finding best " + str(j + 1) + " users..")
                # prepare data structures
                accid = []
                te_r2 = []
                tr_r2 = []
                sizes = []

                # iterate over all account ids
                for k in self.accids:

                    # prepare data structures
                    te_r2_l = []
                    tr_r2_l = []

                    if k in acc_so_far:
                        accid.append(str(k))
                        te_r2.append(0)
                        sizes.append(0)
                        tr_r2.append(0)
                    else:
                        if len(acc_so_far) == 0:
                            accountid = [k]
                        else:
                            accountid = acc_so_far + [k]

                        # subset data
                        accid.append(str(k))
                        user_data = self.data[self.data['Account'].isin(accountid)]
                        sizes.append(user_data.shape[0])

                        # prepare data
                        X = user_data[self.indep_var]
                        Y = user_data[[self.dep_var]].values.ravel()

                        # repeat 20 times
                        for l in range(20):
                            # split into 80/20 training and testing
                            Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2,
                                                                            stratify=user_data['Account'])
                            # fit the model
                            self.models[m].fit(Xtrain, Ytrain)

                            # calculate r2 on training data
                            pred_train = self.models[m].predict(Xtrain)
                            tr_r2_l.append(r2_score(pred_train, Ytrain))

                            # calculate r2 on testing data
                            pred_test = self.models[m].predict(Xtest)
                            te_r2_l.append(r2_score(pred_test, Ytest))

                        # save mean r2
                        te_r2_l = [i if i > 0 else 0 for i in te_r2_l]
                        tr_r2_l = [i if i > 0 else 0 for i in tr_r2_l]
                        te_r2.append(numpy.mean(te_r2_l))
                        tr_r2.append(numpy.mean(tr_r2_l))

                # get rankings to decide best kth account to add to acc_so_far
                r_tr_r2 = self.Getranking(tr_r2, accid, order=1)
                r_te_r2 = self.Getranking(te_r2, accid, order=1)
                r_size = self.Getranking(sizes, accid, order=1)
                ranks_cl = pd.DataFrame({'Name': list(r_te_r2.keys()),
                                         'Ranking': [(x1 + x2 + x3) / 3 for x1, x2, x3 in
                                                     zip(r_tr_r2.values(),
                                                         r_te_r2.values(),
                                                         r_size.values())]})
                ranks_cl = ranks_cl.sort_values('Ranking', ascending=False)
                accountid = int(ranks_cl[['Name']].values[0][0])
                rank = ranks_cl[['Ranking']].values[0][0]

                if rank == 0:
                    break

                # update entries for j number of accounts
                acc_so_far.append(accountid)
                ind = accid.index(str(accountid))
                best_tr_r2.append(tr_r2[ind])
                best_te_r2.append(te_r2[ind])
                size_so_far.append(sizes[ind])

            # save results in score_collector
            self.score_collector[m]['NumOfUsers'] = range(1, (len(acc_so_far) + 1))
            self.score_collector[m]['Account'] = acc_so_far
            self.score_collector[m]['Size'] = size_so_far
            self.score_collector[m]['TrainR2'] = best_tr_r2
            self.score_collector[m]['TestR2'] = best_te_r2

        return self.score_collector


def BuildMARM():
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

    # process methodnames
    methodnames = args.methodnames.split(",")

    # begin building MARMs
    print("Starting to build mixed account regression models (MARMs).")
    start_time = time.time()

    # save independent variables and dependent variables
    pd.DataFrame({'IndepVars': indep_vars}).to_csv(args.output_dir + "/" + args.sys_name + "_" + args.dep_vars +
                                                   "_" + "IndepVars.csv")

    bmarm = BuildMixedAccountModels(path_to_data=args.path_to_data, methodnames=methodnames,
                                    min_num_jobs=args.min_num_jobs,
                                    indep_var=indep_vars, dep_var=args.dep_vars)
    scores = bmarm.MARM()

    end_time = time.time()
    print("MARMs built and evaluated in " + str(round(end_time - start_time, 2)) + " seconds.")

    # save account info
    bmarm.accinfo.to_csv(args.output_dir + "/" + args.sys_name + "_accinfo.csv")

    # save report
    for i in range(len(scores)):
        scores[i].to_csv(args.output_dir + "/" + args.sys_name + "_" + args.dep_vars + "_"
                         + methodnames[i] + "_MARM_report.csv")

    # plot report
    PlotReport(args.output_dir, args.dep_vars, args.sys_name, methodnames)


def PlotReport(output_dir, yvar, sys_name, methodnames):
    # read each report
    for i in range(len(methodnames)):
        scores = pd.read_csv(output_dir + "/" + sys_name + "_" + yvar + "_"
                             + methodnames[i] + "_MARM_report.csv")
        scores["NumOfUsers"] = range(1, scores.shape[0]+1)
        # write custom xticks
        num_jobs = scores['Size'] / 1000
        num_jobs = numpy.round(num_jobs)
        num_jobs = list(map(str, list(map(int, num_jobs))))
        num_jobs = [str(n + 1) + " (" + num_jobs[n] + "K)" for n in range(len(num_jobs))]

        # plot
        plt.rc('xtick', labelsize=9)
        scores.plot.line(x='NumOfUsers', y=['TrainR2', 'TestR2'], color=['red', 'orange'])
        plt.grid(color='gray', linewidth=0.1)
        plt.xticks(scores['NumOfUsers'], labels=num_jobs, rotation='vertical')
        plt.title(sys_name + " " + yvar + " MARM based on" + "\n" + methodnames[i])
        plt.xlabel("# of users (# of jobs)")
        plt.ylabel("R2")
        plt.tight_layout()
        plt.savefig(output_dir + "/" + sys_name + "_" + yvar + "_"
                    + methodnames[i] + "_MARM_report.pdf")
        plt.close()


def setup_arguments():
    arguments = argparse.ArgumentParser()
    arguments.add_argument('-path_to_data', help='string; Full path to cleaned data obtained using PreProcess.',
                           required=True, type=str)
    arguments.add_argument('-output_dir', help='string; Path to output directory to save results. If such a directory '
                                               'does not exist, it will be created.', required=True, type=str)
    arguments.add_argument('-sys_name', help='string; Name of the HPC System.', required=True, type=str)
    arguments.add_argument('-dep_vars', help='string; Name of the response variable to regress '
                                             '(options include: CPUTimeRAW or MaxRSS).', required=True, type=str)
    arguments.add_argument('-indep_vars', help='string; A comma (,) separated name of factors to be used in '
                                               'building the regression models '
                                               '(default: Timelimit,ReqCPUS,ReqMem,ReqNodes,QOS).'
                           , required=False, default="Timelimit,ReqCPUS,ReqMem,ReqNodes,QOS", type=str)
    arguments.add_argument('-methodnames', help='string; A comma (,) separated name of ML models to be used to '
                                                'build MARMs. Valid options include any combination of the following '
                                                'regression models: LR, Ridge, LassoLARS, ElasticNet, RandomForest, '
                                                'CART and LightGBM (default: LightGBM,RandomForest,CART).',
                           required=False, type=str, default="LightGBM,RandomForest,CART")
    arguments.add_argument('-min_num_jobs', help='int; The minimum number of jobs under an account for it to be '
                                                 'considered for single and mixed account modeling (default: 1000).',
                           required=False, type=int, default=1000)

    return arguments


BuildMARM()
