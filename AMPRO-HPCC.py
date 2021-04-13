import joblib
import argparse
import numpy
import re
import sys


def AMPRO_HPCC():

    # setup args list
    arguments = setup_arguments()
    args = arguments.parse_args()

    sh_file_name = args.i

    f = open(sh_file_name, "r")
    f = f.readlines()

    case = ExtractFields(f)
    case = numpy.asarray(list(case.values()))

    if case[0] == -1 or case[3] == -1:
        sys.exit()

    if args.f is 1:

        fail_status = PredictFailure(case, args)
        succ_prob = fail_status[1][0][0] * 100
        if succ_prob >= 60:
            print("Your job has " + str(round(succ_prob, 1)) + "% chance of success.")
        else:
            print("Your job has " + str(round((100 - succ_prob), 1)) +
                  "% chance of failure, consider changing the requirements.")

    if args.t is 1:

        pred_time = PredictTime(case, args)
        if pred_time == case[0]:
            print("No adjustment required for time-limit.")
        else:
            print("Timelimit recommendation : "+str(pred_time[0])+" hrs")

    if args.m is 1:

        pred_mem = PredictMem(case, args)
        if pred_mem == case[2]:
            print("No adjustment required for total memory.")
        else:
            print("Total memory recommendation : "+str(pred_mem*1024)+" MB")
            print("Memory required per CPU recommendation : "+str((pred_mem*1024)/case[3])+" MB")


# Use regex to extract important values
def ExtractFields(f):

    pattern = "\\#SBATCH[ |\t]--([a-zA-Z\\-]+)[=| ]([0-9a-zA-Z\\.\\%\\:\\-@]+)[ |\n]"
    param = []
    values = []
    for i in f:
        s = re.match(pattern, i)
        if s:
            param.append(s.group(1))
            values.append(s.group(2))

    timel = ConvertTime(values[param.index("time")])
    qos = values[param.index("qos")]
    nodes = int(values[param.index("nodes")])
    if "ntasks" in param:
        ncpus = int(values[param.index("ntasks")])
    elif "ntasks-per-node" in param:
        ncpus = int(values[param.index("ntasks-per-node")]) * nodes
    else:
        ncpus = -1
        print("Could not find any parameter specifying ncpus. Please specify ntasks or ntasks-per-node.")

    memtot = ConvertMem(values[param.index("mem-per-cpu")]) * ncpus

    case = {
        "Timelimit": timel,
        "ReqMem": memtot / ncpus,
        "ReqMemTotal": memtot,
        "ReqCPUS": ncpus,
        "QOS": qos,
        "ReqNodes": nodes
    }

    return case


# Convert given time format day-hr:min:sec into hours
def ConvertTime(time):

    pattern = "([0-9]+)\\-([0-9]+)\\:([0-9]+)\\:([0-9]+)"
    matches = re.match(pattern, time)

    if not matches:
        print("No valid time-limit specification found. Please make sure your time-limit specification follows"
              " dd-hh:mm:ss format")
        time_in_hrs = -1
    else:
        time_in_hrs = (int(matches.group(1))*24) + int(matches.group(2)) + (int(matches.group(3))/60) + \
                      (int(matches.group(4))/3600)

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
    

# Predict failure
def PredictFailure(case, args):

    case_log = case.copy()
    for i in range(0, case_log.shape[0]):
        case_log[i] = numpy.log1p(case_log[i])

    std_sc = joblib.load(args.mod + "failed_std_sc_transform.pkl")
    case_log = std_sc.transform(case_log.reshape(1, -1))

    model = joblib.load(args.mod + "rf_failure_classification.pkl")
    pred_st = model.predict(case_log)
    pred_prob = model.predict_proba(case_log)

    return [pred_st, pred_prob]


# Predict time
def PredictTime(case, args):

    case_log = case.copy()

    reqY = case[0]

    std_sc = joblib.load(args.mod + "req_time_std_sc_transform.pkl")
    case_log = std_sc.transform(case_log.reshape(1, -1))

    model = joblib.load(args.mod + "rf_req_time_regression.pkl")
    pred = model.predict(case_log)
    pred = numpy.ceil(pred + (0.3 * pred))
    
    if pred > reqY:
        pred = reqY

    return pred


# Predict mem
def PredictMem(case, args):

    case_log = case.copy()

    reqY = case[2]

    std_sc = joblib.load(args.mod + "req_mem_std_sc_transform.pkl")
    case_log = std_sc.transform(case_log.reshape(1, -1))

    model = joblib.load(args.mod + "rf_req_mem_regression.pkl")
    pred = model.predict(case_log)

    # round up to nearest 'ncpus' MB
    base = case[3] / 1024
    pred = base * numpy.ceil(round(pred[0]/base))

    if pred > reqY:
        pred = reqY

    return pred


# setup argument names and help options
def setup_arguments():

    arguments = argparse.ArgumentParser()
    arguments.add_argument('-path_to_script', help='string; Full path to the slurm script file.',
                           required=True, type=str)
    arguments.add_argument('-path_to_model', help='string; Full path to the .pkl MARM model, if not provided ML models '
                                                  'will be searched in the current directory.',
                           required=False, type=str, default="./")
    arguments.add_argument('-path_to_defaults', help='string; Full path to the .csv system_defaults file, if not '
                                                     'provided ML models will be searched in the current directory.',
                           required=False, type=str, default="./")
    arguments.add_argument('-path_to_stdscale', help='string; Full path to the .sav standard scalar transform, if not '
                                                     'provided ML models will be searched in the current directory.',
                           required=False, type=str, default="./")
    return arguments


AMPRO_HPCC()
