import logging
import os
import sys
import time
import timeit

import numpy as np

from ..algorithms.qsa import QSA
from ..config import DEFAULT_CONFIG
from ..data.synthetic import data_split, get_data
from ..models.logistic_regression import eval_ghat, f_hat, simple_logistic

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "exp/exp_{}/bin/"


def store_result(
    theta,
    theta1,
    testX,
    testY,
    testT,
    passedSafetyTest,
    worker_id,
    nWorkers,
    m,
    trial,
    numTrials,
    seldonian_type,
    ls_dumb,
    config=DEFAULT_CONFIG,
):
    """
    Print and store the resultant information in a file.

    :param theta: The parameters of the model
    :param theta1: The additional parameter of the model, often the last parameter
    :param testX: The features of the test dataset
    :param testY: The labels of the test dataset
    :param testT: The sensitive attribute column of the test dataset
    :param passedSafetyTest: Whether the safety test was passed
    :param worker_id: Id of the worker thread
    :param nWorkers: Total number of worker threads
    :param trial: Trial number of the experiment on the worker thread
    :param numTrials: Total number of trials
    :param seldonian_type: Mode used in the experiment
    :return: (solution_found, failure_g, upper_bound, fhat) tuple values
    """
    if ls_dumb:
        trueLogLoss = float(-f_hat(theta, theta1, testX, testY))
        upper_bound = float(
            eval_ghat(theta, theta1, testX, testY, testT, seldonian_type, config)
        )
        failures_g1 = 0
        if upper_bound > 0:
            failures_g1 = 1
        w = f"(worker {worker_id}/{nWorkers})"
        t = f"trial {trial + 1}/{numTrials}"
        logger.info(
            "[%s %s %s, m %s] f_hat: %.10f, upper bound: %.10f",
            w,
            ls_dumb,
            t,
            m,
            trueLogLoss,
            upper_bound,
        )
        return 1, failures_g1, upper_bound, -trueLogLoss
    elif passedSafetyTest:
        trueLogLoss = float(-f_hat(theta, theta1, testX, testY))
        u = float(eval_ghat(theta, theta1, testX, testY, testT, seldonian_type, config))
        failures_g1 = 0
        if u > 0:
            failures_g1 = 1
        w = f"(worker {worker_id}/{nWorkers})"
        t = f"trial {trial + 1}/{numTrials}"
        logger.info(
            "[%s %s %s, m %s] Solution: [%s, %s] f_hat: %.10f, upper bound: %.10f",
            w,
            seldonian_type,
            t,
            m,
            theta,
            theta1,
            trueLogLoss,
            u,
        )
        return 1, failures_g1, u, -trueLogLoss
    else:
        w = f"(worker {worker_id}/{nWorkers})"
        t = f"trial {trial + 1}/{numTrials}"
        logger.info("[%s SBase %s, m %s] No solution found", w, t, m)
        return 0, 0, 0, None


def run_experiments(
    worker_id,
    nWorkers,
    ms,
    numM,
    numTrials,
    mTest,
    N,
    seldonian_type,
    config=DEFAULT_CONFIG,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    """
    Main function that runs the experiment.

    :param worker_id: Id of the worker thread
    :param nWorkers: Total number of worker threads
    :param ms: Array containing the fraction values of the amount of data to be used
    :param numM: Number of fractions of data
    :param numTrials: Total number of trials
    :param mTest: The fraction of test samples to be used from the complete dataset
    :param N: Number of data samples of the synthetic dataset
    :param seldonian_type: Mode used in the experiment
    :return: None
    """
    s_solutions_found = np.zeros((numTrials, numM))
    s_failures_g1 = np.zeros((numTrials, numM))
    s_upper_bound = np.zeros((numTrials, numM))
    s_fs = np.zeros((numTrials, numM))

    LS_solutions_found = np.zeros((numTrials, numM))
    LS_failures_g1 = np.zeros((numTrials, numM))
    LS_upper_bound = np.zeros((numTrials, numM))
    LS_fs = np.zeros((numTrials, numM))

    experiment_number = worker_id
    output_path = output_dir.format(seldonian_type)
    outputFile = os.path.join(output_path, "results%d.npz" % experiment_number)
    logger.info("Writing output to %s", outputFile)

    base_seed = (experiment_number * 99) + 1
    All = get_data(N, 5, 0.4, 0.4, 0.6, base_seed)
    init_sol, init_sol1 = None, None

    for trial in range(numTrials):
        for mIndex, m in enumerate(ms):
            base_seed = (experiment_number * numTrials) + 1
            random_state = base_seed + trial
            testX, testY, testT, trainX, trainY, trainT = data_split(
                m, All, random_state, mTest
            )

            theta, theta1 = simple_logistic(trainX, trainY)
            (
                LS_solutions_found[trial, mIndex],
                LS_failures_g1[trial, mIndex],
                LS_upper_bound[trial, mIndex],
                LS_fs[trial, mIndex],
            ) = store_result(
                theta,
                theta1,
                testX,
                testY,
                testT,
                True,
                worker_id,
                nWorkers,
                m,
                trial,
                numTrials,
                seldonian_type,
                "LS",
                config,
            )

            (theta, theta1, passedSafetyTest) = QSA(
                trainX, trainY, trainT, seldonian_type, init_sol, init_sol1, config
            )
            (
                s_solutions_found[trial, mIndex],
                s_failures_g1[trial, mIndex],
                s_upper_bound[trial, mIndex],
                s_fs[trial, mIndex],
            ) = store_result(
                theta,
                theta1,
                testX,
                testY,
                testT,
                passedSafetyTest,
                worker_id,
                nWorkers,
                m,
                trial,
                numTrials,
                seldonian_type,
                None,
                config,
            )
            if s_solutions_found[trial, mIndex] == 1:
                init_sol, init_sol1 = theta, theta1

    np.savez(
        outputFile,
        ms=ms,
        s_solutions_found=s_solutions_found,
        s_fs=s_fs,
        s_failures_g1=s_failures_g1,
        s_upper_bound=s_upper_bound,
        LS_solutions_found=LS_solutions_found,
        LS_fs=LS_fs,
        LS_failures_g1=LS_failures_g1,
        LS_upper_bound=LS_upper_bound,
    )
    logger.info("Saved the file %s", outputFile)


if __name__ == "__main__":
    import logging

    import ray  # pyrefly: ignore

    logging.basicConfig(filename="runner.log", level=logging.INFO)
    ray.init()

    run_experiments_remote = ray.remote(run_experiments)

    logger.info("Assuming the default: 50")
    nWorkers = 2
    logger.info("Running experiments on %d threads", nWorkers)
    N = 10000
    ms = np.logspace(-2, 0, num=3)
    logger.info("N %d, frac array: %s", N, ms)
    logger.info("Running for: %s", sys.argv[1])
    numM = len(ms)
    numTrials = 2
    mTest = 0.2
    logger.info("Number of trials: %d", numTrials)

    tic = timeit.default_timer()
    _ = ray.get(
        [
            run_experiments_remote.remote(
                worker_id, nWorkers, ms, numM, numTrials, mTest, N, sys.argv[1]
            )
            for worker_id in range(1, nWorkers + 1)
        ]
    )
    toc = timeit.default_timer()
    time_parallel = toc - tic
    logger.info("Time elapsed: %s", time_parallel)
    time.sleep(2)
    ray.shutdown()
