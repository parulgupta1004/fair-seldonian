from __future__ import annotations

import logging
import os
import sys
import time
import timeit
from typing import TYPE_CHECKING

import numpy as np

from ..algorithms.qsa import QSA
from ..config import DEFAULT_CONFIG, SeldonianConfig
from ..data.synthetic import data_split, get_data
from ..models.logistic_regression import eval_ghat, f_hat, simple_logistic

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "exp/exp_{}/bin/"


def store_result(
    theta: torch.Tensor,
    theta1: torch.Tensor,
    test_x: np.ndarray,
    test_y: np.ndarray,
    test_t: np.ndarray,
    passed_safety_test: bool,
    worker_id: int,
    n_workers: int,
    m: float,
    trial: int,
    num_trials: int,
    seldonian_type: str,
    is_baseline: bool,
    config: SeldonianConfig = DEFAULT_CONFIG,
) -> tuple[int, int, float, float | None]:
    """
    Print and store the resultant information in a file.

    :param theta: The parameters of the model
    :param theta1: The additional parameter of the model, often the last parameter
    :param test_x: The features of the test dataset
    :param test_y: The labels of the test dataset
    :param test_t: The sensitive attribute column of the test dataset
    :param passed_safety_test: Whether the safety test was passed
    :param worker_id: Id of the worker thread
    :param n_workers: Total number of worker threads
    :param trial: Trial number of the experiment on the worker thread
    :param num_trials: Total number of trials
    :param seldonian_type: Mode used in the experiment
    :param is_baseline: Whether this is the unconstrained logistic-regression baseline
    :return: (solution_found, failure_g, upper_bound, fhat) tuple values
    """
    worker = f"(worker {worker_id}/{n_workers})"
    trial_label = f"trial {trial + 1}/{num_trials}"

    if not is_baseline and not passed_safety_test:
        logger.info(f"[{worker} SBase {trial_label}, m {m}] No solution found")
        return 0, 0, 0, None

    # Only scalars are needed below; detach so float() doesn't warn / retain the graph.
    theta, theta1 = theta.detach(), theta1.detach()
    true_log_loss = float(-f_hat(theta, theta1, test_x, test_y))
    upper_bound = float(
        eval_ghat(theta, theta1, test_x, test_y, test_t, seldonian_type, config)
    )
    failures_g1 = 1 if upper_bound > 0 else 0

    if is_baseline:
        logger.info(
            f"[{worker} LS {trial_label}, m {m}] "
            f"f_hat: {true_log_loss:.10f}, upper bound: {upper_bound:.10f}"
        )
    else:
        logger.info(
            f"[{worker} {seldonian_type} {trial_label}, m {m}] "
            f"Solution: [{theta}, {theta1}] "
            f"f_hat: {true_log_loss:.10f}, upper bound: {upper_bound:.10f}"
        )
    return 1, failures_g1, upper_bound, -true_log_loss


def run_experiments(
    worker_id: int,
    n_workers: int,
    ms: np.ndarray,
    num_trials: int,
    m_test: float,
    N: int,
    seldonian_type: str,
    config: SeldonianConfig = DEFAULT_CONFIG,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> None:
    """
    Main function that runs the experiment.

    :param worker_id: Id of the worker thread
    :param n_workers: Total number of worker threads
    :param ms: Array containing the fraction values of the amount of data to be used
    :param num_trials: Total number of trials
    :param m_test: The fraction of test samples to be used from the complete dataset
    :param N: Number of data samples of the synthetic dataset
    :param seldonian_type: Mode used in the experiment
    :return: None
    """
    num_m = len(ms)
    s_solutions_found = np.zeros((num_trials, num_m))
    s_failures_g1 = np.zeros((num_trials, num_m))
    s_upper_bound = np.zeros((num_trials, num_m))
    s_fs = np.zeros((num_trials, num_m))

    LS_solutions_found = np.zeros((num_trials, num_m))
    LS_failures_g1 = np.zeros((num_trials, num_m))
    LS_upper_bound = np.zeros((num_trials, num_m))
    LS_fs = np.zeros((num_trials, num_m))

    experiment_number = worker_id
    output_path = output_dir.format(seldonian_type)
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"results{experiment_number}.npz")
    logger.info(f"Writing output to {output_file}")

    base_seed = (experiment_number * 99) + 1
    all_data = get_data(N, 5, 0.4, 0.4, 0.6, base_seed)
    init_sol: torch.Tensor | None = None
    init_sol1: torch.Tensor | None = None

    for trial in range(num_trials):
        for m_index, m in enumerate(ms):
            base_seed = (experiment_number * num_trials) + 1
            random_state = base_seed + trial
            test_x, test_y, test_t, train_x, train_y, train_t = data_split(
                m, all_data, random_state, m_test
            )

            theta, theta1 = simple_logistic(train_x, train_y)
            (
                LS_solutions_found[trial, m_index],
                LS_failures_g1[trial, m_index],
                LS_upper_bound[trial, m_index],
                LS_fs[trial, m_index],
            ) = store_result(
                theta,
                theta1,
                test_x,
                test_y,
                test_t,
                True,
                worker_id,
                n_workers,
                m,
                trial,
                num_trials,
                seldonian_type,
                True,
                config,
            )

            theta, theta1, passed_safety_test = QSA(
                train_x, train_y, train_t, seldonian_type, init_sol, init_sol1, config
            )
            (
                s_solutions_found[trial, m_index],
                s_failures_g1[trial, m_index],
                s_upper_bound[trial, m_index],
                s_fs[trial, m_index],
            ) = store_result(
                theta,
                theta1,
                test_x,
                test_y,
                test_t,
                passed_safety_test,
                worker_id,
                n_workers,
                m,
                trial,
                num_trials,
                seldonian_type,
                False,
                config,
            )
            if s_solutions_found[trial, m_index] == 1:
                init_sol, init_sol1 = theta, theta1

    np.savez(
        output_file,
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
    logger.info(f"Saved the file {output_file}")


if __name__ == "__main__":
    import logging

    import ray  # pyrefly: ignore

    logging.basicConfig(filename="runner.log", level=logging.INFO)
    ray.init()

    run_experiments_remote = ray.remote(run_experiments)

    logger.info("Assuming the default: 50")
    n_workers = 2
    logger.info(f"Running experiments on {n_workers} threads")
    N = 10000
    ms = np.logspace(-2, 0, num=3)
    logger.info(f"N {N}, frac array: {ms}")
    logger.info(f"Running for: {sys.argv[1]}")
    num_trials = 2
    m_test = 0.2
    logger.info(f"Number of trials: {num_trials}")

    tic = timeit.default_timer()
    _ = ray.get(
        [
            run_experiments_remote.remote(
                worker_id, n_workers, ms, num_trials, m_test, N, sys.argv[1]
            )
            for worker_id in range(1, n_workers + 1)
        ]
    )
    toc = timeit.default_timer()
    time_parallel = toc - tic
    logger.info(f"Time elapsed: {time_parallel}")
    time.sleep(2)
    ray.shutdown()
