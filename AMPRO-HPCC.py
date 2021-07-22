# import libraries
import joblib
import argparse
import numpy
import re
import os
import pandas as pd


def AMPRO_HPCC():

    # setup args list
    arguments = setup_arguments()
    args = arguments.parse_args()

    # check if the path_to_script is valid
    if not os.path.isfile(args.path_to_script):
        print("Invalid path to script! Exiting..")
        raise SystemExit

    # check path to utilities
    if not os.path.isdir(args.path_to_util):
        print("Invalid path to utility directory! Exiting..")
        raise SystemExit

    # load independent variables
    indep_vars_time = pd.read_csv(args.path_to_util + "/" + args.sys_name + "_CPUTimeRAW_IndepVars.csv")
    indep_vars_time = indep_vars_time['IndepVars'].values
    indep_vars_mem = pd.read_csv(args.path_to_util + "/" + args.sys_name + "_MaxRSS_IndepVars.csv")
    indep_vars_mem = indep_vars_mem['IndepVars'].values

    # load standard scalar transform
    stdscale = joblib.load(args.path_to_util + "/" + args.sys_name + "_StandardScalarModel.sav")

    # load ML model
    model_time = joblib.load(args.path_to_util + "/" + args.sys_name + "_CPUTimeRAW_MARM.pkl")
    model_maxrss = joblib.load(args.path_to_util + "/" + args.sys_name + "_MaxRSS_MARM.pkl")

    # load system defaults
    sysdef = pd.read_csv(args.path_to_util + "/" + args.sys_name + "_" + "SysDefault.csv")

    # extract data from script
    sh_file_name = args.path_to_script

    f = open(sh_file_name, "r")
    f = f.readlines()

    case = ExtractFields(f, sysdef)

    # predict time and memory
    case_log = case.copy()

    # fix QOS
    qos_factors = pd.read_csv(args.path_to_util + "/" + args.sys_name + "QOSHash_Factors.csv")
    qos_factors = qos_factors['QOSFactor'].values[numpy.where(qos_factors['QOS'] == case_log['QOS'])[0]][0]
    case_log['QOS'] = qos_factors

    # requested time and memory
    req_time = case_log['Timelimit'].values[0]
    req_mem = case_log['ReqMem'].values[0]

    # standard scalar transform
    case_log_tr = stdscale.transform(case_log)
    case_log = pd.DataFrame(case_log_tr, index=range(case_log.shape[0]),
                            columns=case_log.columns)
    # time
    # get independent variables
    case_log_time = case_log[indep_vars_time]

    # predict
    pred_time = model_time.predict(case_log_time)
    pred_time = numpy.ceil(pred_time + (0.3 * pred_time))

    # memory
    # get independent variables
    case_log_mem = case_log[indep_vars_mem]

    # predict
    pred_mem = model_maxrss.predict(case_log_mem)
    pred_mem = numpy.ceil(pred_mem + (0.3 * pred_mem))

    # reverse transform
    case_log['CPUTimeRAW'][0] = pred_time * 3600
    case_log['MaxRSS'][0] = pred_mem
    case_rev_log = stdscale.inverse_transform(case_log)
    case_rev_log = pd.DataFrame(case_rev_log, index=range(case_log.shape[0]),
                                columns=case_log.columns)
    # fix time and memory if needed
    if case_rev_log['CPUTimeRAW'].values[0] > req_time:
        case_rev_log['CPUTimeRAW'][0] = req_time

    if case_rev_log['MaxRSS'].values[0] > req_mem:
        case_rev_log['MaxRSS'][0] = req_mem

    print("Timelimit recommendation : " + str(case_rev_log['CPUTimeRAW'].values[0]) + " hrs")
    print("Memory recommendation : "+str(case_rev_log['MaxRSS'].values[0]*1024)+" MB")


# Use regex to extract important values
def ExtractFields(f, sys_def):

    pattern = "\\#SBATCH[ |\t]--([a-zA-Z\\-]+)[=| ]([0-9a-zA-Z\\.\\%\\:\\-@]+)[ |\n]"
    param = []
    values = []
    for i in f:
        s = re.match(pattern, i)
        if s:
            param.append(s.group(1))
            values.append(s.group(2))

    if "time" in param:
        timel = ConvertTime(values[param.index("time")])
    else:
        timel = sys_def['Time'].values[0]

    if "qos" in param:
        qos = values[param.index("qos")]
    else:
        qos = sys_def['QOS'].values[0]

    if "nodes" in param:
        nodes = int(values[param.index("nodes")])
    else:
        nodes = 1

    if "ntasks" in param:
        ncpus = int(values[param.index("ntasks")])
    elif "ntasks-per-node" in param:
        ncpus = int(values[param.index("ntasks-per-node")]) * nodes
    else:
        ncpus = -1
        print("Could not find any parameter specifying ncpus. Please specify ntasks or ntasks-per-node.")

    if "mem" in param:
        memtot = ConvertMem(values[param.index("mem")])
    elif "mem-per-cpu" in param:
        memtot = ConvertMem(values[param.index("mem-per-cpu")]) * ncpus
    else:
        memtot = -1
        print("Could not find any parameter specifying memory. Please specify mem or mem-per-cpu.")

    case = pd.DataFrame({
        "Account": 0,
        "AllocCPUS": ncpus,
        "CPUTimeRAW": 0,
        "MaxRSS": 0,
        "ReqCPUS": ncpus,
        "ReqMem": memtot,
        "ReqNodes": nodes,
        "Timelimit": timel,
        "QOS": qos
    }, index=[0])
    return case


# Convert given time format day-hr:min:sec into hours
def ConvertTime(time):

    # try pattern including days
    pattern_id = "([0-9]+)\\-([0-9]+)\\:([0-9]+)\\:([0-9]+)"
    matches_id = re.match(pattern_id, time)

    if not matches_id:
        # trying another pattern excluding days
        pattern_wd = "([0-9]+)\\:([0-9]+)\\:([0-9]+)"
        matches_wd = re.match(pattern_wd, time)

        if not matches_wd:
            print("No valid time-limit specification found, using 24 hrs. Please make sure your time-limit "
                  "specification follows dd-hh:mm:ss or hh:mm:ss format")
            time_in_hrs = 24
        else:
            time_in_hrs = (int(matches_wd.group(1))) + (int(matches_wd.group(2)) / 60) + \
                          (int(matches_wd.group(3)) / 3600)
    else:
        time_in_hrs = (int(matches_id.group(1))*24) + int(matches_id.group(2)) + (int(matches_id.group(3))/60) + \
                      (int(matches_id.group(4))/3600)

    return time_in_hrs


# Convert given memory into GB
def ConvertMem(mem):

    if "K" in mem:
        mem_in_gb = int(mem.replace("K", "")) / 1048576
    elif "M" in mem:
        mem_in_gb = int(mem.replace("M", "")) / 1024
    elif "T" in mem:
        mem_in_gb = int(mem.replace("T", "")) * 1024
    elif "G" in mem:
        mem_in_gb = int(mem.replace("G", ""))
    else:  # Assuming the mem declaration is in MB
        mem_in_gb = int(mem) / 1024

    return mem_in_gb


# setup argument names and help options
def setup_arguments():

    arguments = argparse.ArgumentParser()
    arguments.add_argument('-path_to_script', help='string; Full path to the slurm script file.',
                           required=True, type=str)
    arguments.add_argument('-sys_name', help='string; Name of the HPC System.', required=True, type=str)
    arguments.add_argument('-path_to_util', help='string; Full path to the utility files required for the predictions. '
                                                 'This folder should contain the .pkl MARM model, .csv system '
                                                 'defaults, .csv independent variables and .sav standard scalar '
                                                 'transform. If not provided the files will be searched in the '
                                                 'current directory.', required=False, type=str, default="./")
    return arguments


AMPRO_HPCC()
