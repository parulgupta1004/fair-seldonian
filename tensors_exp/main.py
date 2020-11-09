import timeit
import numpy as np
import pandas as pd
import ray
import logging
logging.basicConfig( filename= 'main.py', level=logging.INFO )
#ray.shutdown()
ray.init()
from synthetic_data import *
from qsa import *
from logistic_regression_functions import *
import time
# Folder where the experiment results will be saved
bin_path = 'exp/exp_{}/bin/'
import sys

def store_result(theta, theta1, testX, testY, testT, passedSafetyTest,
                 worker_id, nWorkers, m, trial, numTrials, seldonian_type, ls_dumb):
    """
    :param theta:
    :param theta1:
    :param testX:
    :param testY:
    :param testT:
    :param passedSafetyTest:
    :param worker_id:
    :param nWorkers:
    :param trial:
    :param numTrials:
    :param seldonian_type:
    :return: solution_found, failure_g, upper_bound, fhat
    """
    if ls_dumb:
        trueLogLoss = float(-fHat(theta, theta1, testX, testY))
        upper_bound = float(eval_ghat(theta, theta1, testX, testY, testT, seldonian_type))
        failures_g1 = 0
        if upper_bound > 0:
            failures_g1 = 1
        print(f"[(worker {worker_id}/{nWorkers}) {ls_dumb}   trial {trial + 1}/{numTrials}, m {m}]"
              f"{ls_dumb} fHat over test data: {trueLogLoss:.10f}, upper bound: {upper_bound:.10f}")
        return 1, failures_g1, upper_bound, -trueLogLoss
    elif passedSafetyTest:
        trueLogLoss = float(-fHat(theta, theta1, testX, testY))
        u = float(eval_ghat(theta, theta1, testX, testY, testT, seldonian_type))
        failures_g1 = 0
        if u > 0:
            failures_g1 = 1
        print(
            f"[(worker {worker_id}/{nWorkers}) {seldonian_type} trial {trial + 1}/{numTrials}, m {m}]"
            f"Solution found: [{theta}, {theta1}]"
            f"\tfHat over test data: {trueLogLoss:.10f}, upper bound: {u:.10f}")
        return 1, failures_g1, u, -trueLogLoss
    else:
        print(
            f"[(worker {worker_id}/{nWorkers}) SBase trial {trial + 1}/{numTrials}, m {m}]"
            "No solution found")
        return 0, 0, 0, None


@ray.remote
def run_experiments(worker_id, nWorkers, ms, numM, numTrials, mTest, N, seldonian_type):
    # Results of the Seldonian algorithm runs
    s_solutions_found = np.zeros((numTrials, numM))
    s_failures_g1 = np.zeros((numTrials, numM))
    s_upper_bound = np.zeros((numTrials, numM))
    s_fs = np.zeros((numTrials, numM))

    # Results of the logistic regression runs
    LS_solutions_found = np.zeros((numTrials, numM))
    LS_failures_g1 = np.zeros((numTrials, numM))
    LS_upper_bound = np.zeros((numTrials, numM))
    LS_fs = np.zeros((numTrials, numM))

    # Prepares file where experiment results will be saved
    experiment_number = worker_id
    outputFile = bin_path.format(seldonian_type) + 'results%d.npz' % experiment_number
    print("Writing output to", outputFile)

    # Create data
    base_seed = (experiment_number * 99) + 1
    All = get_data(N, 5, 0.4, 0.4, 0.6, base_seed)
    init_sol, init_sol1 = None, None
    trainX, old_train = None, None
    testX, testY, testT, All_train = get_test_data(All, mTest)

    # Generate the data used to evaluate the primary objective and failure rates
    for trial in range(numTrials):
        for (mIndex, m) in enumerate(ms):
            # Generate the training data, D
            base_seed = (experiment_number * numTrials) + 1
            random_state = base_seed + trial
            # these are numpy arrays
            if trainX is not None:
                trainY = trainY.reshape(trainY.shape[0], 1)
                trainT = trainT.reshape(trainT.shape[0], 1)
                old_train = np.concatenate((trainX, trainY, trainT), axis = 1)
            trainX, trainY, trainT, All_train = data_split(m, All_train, random_state, N, pd.DataFrame(old_train))
            # print(trainX.shape, trainY.shape, trainT.shape)
            # Run the logistic regression algorithm- theta, theta1 are tensors
            theta, theta1 = simple_logistic(trainX, trainY)
            LS_solutions_found[trial, mIndex], LS_failures_g1[trial, mIndex], LS_upper_bound[
                trial, mIndex], LS_fs[trial, mIndex] = store_result(theta, theta1,
                                                                    testX, testY, testT,
                                                                    True, worker_id, nWorkers,
                                                                    m, trial, numTrials, seldonian_type, "LS")

            # Run QSA
            (theta, theta1, passedSafetyTest) = QSA(trainX, trainY, trainT, seldonian_type, init_sol, init_sol1)
            s_solutions_found[trial, mIndex], s_failures_g1[trial, mIndex], s_upper_bound[
                trial, mIndex], s_fs[trial, mIndex] = store_result(theta, theta1,
                                                                   testX, testY, testT,
                                                                   passedSafetyTest, worker_id, nWorkers,
                                                                   m, trial, numTrials, seldonian_type,
                                                                   None)
            if s_solutions_found[trial, mIndex] == 1:
                init_sol, init_sol1 = theta, theta1

    np.savez(outputFile, ms = ms,
        s_solutions_found = s_solutions_found,
        s_fs = s_fs,
        s_failures_g1 = s_failures_g1,
        s_upper_bound = s_upper_bound,

        LS_solutions_found = LS_solutions_found,
        LS_fs = LS_fs,
        LS_failures_g1 = LS_failures_g1,
        LS_upper_bound = LS_upper_bound)
    print(f"Saved the file {outputFile}")


if __name__ == "__main__":
    print("Assuming the default: 50")
    nWorkers = 2
    print(f"Running experiments on {nWorkers} threads")
    N = 1000000
    ms = np.logspace(-1, 0, num=3)  # 30 fractions
    print("N {}, frac array: {}".format(N, ms))
    print("Running for: {}".format(sys.argv))
    numM = len(ms)
    numTrials = 2  # 2 * 50 = 100 samples per fraction
    mTest = 0.2  # about 0.2 * 1,000,000 test samples = fraction of total data
    print("Number of trials: ", numTrials)
    # Start 'nWorkers' threads in parallel, each one running 'numTrials' trials. Each thread saves its results to a file
    tic = timeit.default_timer()
    _ = ray.get(
        [run_experiments.remote(worker_id, nWorkers, ms, numM, numTrials, mTest, N, "base") for worker_id in
          range(1, nWorkers + 1)])
    toc = timeit.default_timer()
    time_parallel = toc - tic  # Elapsed time in seconds
    print(f"Time elapsed: {time_parallel}")
    time.sleep(2)
    ray.shutdown()
