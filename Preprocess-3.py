import argparse
import joblib
import math
import os
import re
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class Preprocess:

    # class variables
    data = None
    default_time = None
    default_qos = None
    default_partition = None
    select_partition = None
    select_qos = None
    output_dir = None
    system_name = None

    def __init__(self,
                 data,
                 output_dir,
                 system_name,
                 default_qos,
                 default_time,
                 select_partition,
                 select_qos,
                 default_partition):

        # set class variables
        self.default_time = default_time
        self.default_qos = default_qos
        self.default_partition = default_partition
        self.select_partition = select_partition
        self.select_qos = select_qos
        self.output_dir = output_dir
        self.system_name = system_name

        # check if data is a pandas df
        if not isinstance(data, pd.DataFrame):
            raise Exception('data should be a pandas dataframe!')

        # take selected columns in data
        else:
            self.data = data[['Account', 'AllocCPUS', 'CPUTimeRAW', 'MaxRSS', 'ReqCPUS',
                              'ReqMem', 'ReqNodes', 'Timelimit', 'Partition', 'QOS', 'State']]

    def process_data(self):

        pd.options.mode.chained_assignment = None  # prevent false positive warnings

        # fix and modify columns
        self.fix_data()
        self.modify_columns()

        # remove State and Partition from data
        self.data = self.data[['Account', 'AllocCPUS', 'CPUTimeRAW', 'MaxRSS', 'ReqCPUS',
                              'ReqMem', 'ReqNodes', 'Timelimit', 'QOS']]

        # save account
        acc = self.data['Account']
        qos = self.data['QOS']

        # standard scalar transform
        stdscale_model = StandardScaler()
        std_scaled_data = stdscale_model.fit_transform(self.data.values)
        self.data = pd.DataFrame(std_scaled_data,
                                 index=range(self.data.shape[0]),
                                 columns=self.data.columns)

        # save standard scalar model
        joblib.dump(stdscale_model, self.output_dir + "/" + self.system_name + "_StandardScalarModel.sav")

        self.data['Account'] = acc
        self.data['QOS'] = qos

        return self.data

    def fix_data(self):

        # fix default time, QOS and partition
        self.data['Timelimit'][self.data['Timelimit'].isnull()] = self.default_time
        self.data['QOS'][self.data['QOS'].isnull()] = self.default_qos

        if self.default_partition is None:
            self.data['Partition'][self.data['Partition'].isnull()] = self.system_name + "default"
        else:
            self.data['Partition'][self.data['Partition'].isnull()] = self.default_partition

        # fix AllocCPUS, MaxRSS and CPUTimeRAW
        self.data['MaxRSS'][self.data['MaxRSS'].isnull()] = "0"  # MaxRSS is currently a string
        self.data['AllocCPUS'][self.data['AllocCPUS'].isnull()] = 0
        self.data['CPUTimeRAW'][self.data['CPUTimeRAW'].isnull()] = 0

        # fix selected qos and selected partition
        if self.select_qos is None:
            self.select_qos = self.data['QOS']

        if self.select_partition is None:
            self.select_partition = self.data['Partition']

        # filter data by dropping rows
        # QOS = select_qos only
        # Partition = select_partition only
        # MaxRSS = !NA and !zero values only
        # CPUTimeRAW = !NA and !zero values only
        # AllocCPUS = !NA and !zero values only
        # State = 'COMPLETED'
        # Timelimt = !UNLIMITED
        drop_rows = [np.where((~self.data['QOS'].isin(self.select_qos)) |
                              (~self.data['Partition'].isin(self.select_partition)) |
                              (self.data['MaxRSS'] == "0") |
                              (self.data['CPUTimeRAW'] == 0) |
                              (self.data['AllocCPUS'] == 0) |
                              (self.data['State'] != 'COMPLETED') |
                              (self.data['Timelimit'] == 'UNLIMITED'))]

        drop_rows = np.concatenate(drop_rows[0])

        self.data = self.data.drop(drop_rows, axis=0)
        self.data = self.data.reset_index()

        # Prepare 'Account' factors
        acc_factors = pd.factorize(self.data.Account)[0]
        pd.DataFrame({'AccountHash': self.data['Account'],
                      'AccountFactor': acc_factors}). \
            to_csv(self.output_dir + "/" + self.system_name + "AccountHash_Factors.csv")
        self.data['Account'] = acc_factors

        # Prepare 'QOS' factors
        qos_factors = pd.factorize(self.data['QOS'])[0]
        pd.DataFrame({'QOS': self.data['QOS'],
                      'QOSFactor': qos_factors}). \
            to_csv(self.output_dir + "/" + self.system_name + "QOSHash_Factors.csv")
        self.data['QOS'] = qos_factors

    def stdMem(self, x):
        # standardize memory to GB

        try:
            x = float(x)
            if math.isnan(x):
                return np.nan
            else:
                return x
        except:
            if 'M' in x or 'm' in x:
                return float(re.sub(r'M|m', '', x)) / 1024
            elif 'G' in x or 'g' in x:
                return float(re.sub(r'G|g', '', x))
            elif 'K' in x or 'k' in x:
                return float(re.sub(r'K|k', '', x)) / (1024 ** 2)
            elif 'T' in x or 't' in x:
                return float(re.sub(r'T|t', '', x)) * 1024
            else:
                return np.nan

    def stdReqMem(self, rm, rc, rn):
        # standardize required memory to GB normalizing to number of nodes / cpu
        nrm = []
        for i in range(rm.shape[0]):
            m = rm.loc[i]
            c = rc.loc[i]
            n = rn.loc[i]
            try:
                x = float(m)
                if math.isnan(x):
                    nrm.append(np.nan)
                else:
                    nrm.append(x)
            except:
                if 'M' in m or 'm' in m:
                    x = re.sub(r'M|m', '', m)
                    if 'c' in x:
                        nrm.append((float(re.sub(r'c', '', x)) / 1024) * int(c))
                    else:
                        nrm.append((float(re.sub(r'n', '', x)) / 1024) * int(n))

                elif 'G' in m or 'g' in m:
                    x = re.sub(r'G|g', '', m)
                    if 'c' in x:
                        nrm.append(float(re.sub(r'c', '', x)) * int(c))
                    else:
                        nrm.append(float(re.sub(r'n', '', x)) * int(n))

                elif 'K' in m or 'k' in m:
                    x = re.sub(r'K|k', '', m)
                    if 'c' in x:
                        nrm.append(((float(re.sub(r'c', '', x)) / (1024 ** 2)) * int(c)))
                    else:
                        nrm.append(((float(re.sub(r'n', '', x)) / (1024 ** 2)) * int(n)))

                elif 'T' in m or 't' in m:
                    x = re.sub(r'T|t', '', m)
                    if 'c' in x:
                        nrm.append((float(re.sub(r'c', '', x)) * 1024) * int(c))
                    else:
                        nrm.append((float(re.sub(r'n', '', x)) * 1024) * int(n))
                else:
                    nrm.append(np.nan)
        return nrm

    def stdTime(self, x):
        # standardize time to hours
        try:
            x = float(x)
            if math.isnan(x):
                return np.nan
            else:
                return x
        except:
            t = 0

            # deal with days
            if '-' in x:
                x1 = x.split('-')
                t = t + (float(x1[0]) * 24)
                x1 = x1[1]
            else:
                x1 = x

            # hours
            x1 = x1.split(':')
            t = t + float(x1[0])
            # minutes
            t = t + (float(x1[1]) / 60)
            # seconds
            if len(x1) == 3:
                t = t + (float(x1[2]) / 3600)
            return t

    def modify_columns(self):

        # modify certain columns to make learning easier
        self.data['MaxRSS'] = self.data.MaxRSS.apply(lambda x: self.stdMem(x))
        self.data['Timelimit'] = self.data.Timelimit.apply(lambda t: self.stdTime(t))
        self.data['ReqMem'] = self.stdReqMem(self.data['ReqMem'], self.data['ReqCPUS'], self.data['ReqNodes'])


def PreprocessData():

    # setup argument list
    arguments = setup_arguments()
    args = arguments.parse_args()

    # check if the provided input path is a file or not
    if not os.path.isfile(args.input_data_path):
        print("Invalid input_data_path. Exiting..")
        raise SystemExit
    # read file
    data = pd.read_csv(args.input_data_path, sep=args.sep, encoding="ISO-8859-1")
    print("Data read complete.")

    # create output directory if it does not exist
    if not os.path.isdir(args.output_dir):
        print("Creating " + args.output_dir)
        os.makedirs(args.output_dir)

    # process sel_partition
    if not args.sel_partition is None:
        sel_partition = args.sel_partition.split(",")
    else:
        sel_partition = None

    # process sel_QOS
    if not args.sel_qos is None:
        sel_qos = args.sel_qos.split(",")
    else:
        sel_qos = None

    print("Beginning pre-processing")
    start_time = time.time()

    preproc_ob = Preprocess(data=data, output_dir=args.output_dir, system_name=args.sys_name,
                            default_time=args.def_time, default_qos=args.def_qos, default_partition=args.def_partition,
                            select_qos=sel_qos, select_partition=sel_partition)

    processed_data = preproc_ob.process_data()

    end_time = time.time()
    print("Pre-processing completed in " + str(end_time - start_time) + "seconds.")

    # save preprocessed data
    joblib.dump(processed_data, args.output_dir + "/" + args.sys_name + "_processed.sav")


def setup_arguments():

    arguments = argparse.ArgumentParser()
    arguments.add_argument('-input_data_path', help='string; Path to job accounting SLURM logs.',
                           required=True, type=str)
    arguments.add_argument('-sep', help='char; String separator at each line for file at "input_data_path" '
                                        '(default: |)', required=False, default="|", type=str)
    arguments.add_argument('-output_dir', help='string; Path to output directory to save cleaned data, as well as '
                                               'other intermediary results. If such a directory does not exist, it '
                                               'will be created.', required=True, type=str)
    arguments.add_argument('-sys_name', help='string; Name of the HPC System.', required=True, type=str)
    arguments.add_argument('-def_qos', help='string; default QOS assignment for the SLURM based HPC system '
                                            '(default: normal). NOTE: Although this is an optional argument, it is '
                                            'highly recommended that a default value is supplied if the default QOS '
                                            'is not "normal" as it is used as a factor for regression.',
                           required=False, type=str, default="normal")
    arguments.add_argument('-def_partition', help='string; default partition assignment for the SLURM based HPC '
                                                  'system (default: name of the system will be used with "default" as'
                                                  'suffix, e.g., if sys_name = "uw", partition would be named'
                                                  '"uwdefault").', required=False, type=str)
    arguments.add_argument('-def_time', help='int; default Timelimit for the SLURM based HPC system in hours '
                                             '(default: 24). NOTE: Although this is an optional argument, it is '
                                             'highly recommended that a default Timelimit is supplied if the default '
                                             'Timelimit is not 24 hours, as it is used as a factor for regression.',
                           required=False, default=24, type=int)
    arguments.add_argument('-sel_partition', help='string; A comma (,) separated name of partitions to keep in '
                                                  'the data, if desired (e.g. if you want to exclude premium '
                                                  'partitions from consideration). All entries with other partitions '
                                                  'will be removed (default: None)', required=False, default=None,
                           type=str)
    arguments.add_argument('-sel_qos', help='string; A comma (,) separated name of QOS to keep in the data, if '
                                            'desired (e.g. if you want to exclude premium QOS from consideration). All '
                                            'entries with other QOS will be removed (default: None)', required=False,
                           default=None, type=str)
    return arguments


PreprocessData()
