import csv
import glob
import logging
import os
import re

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_BIN_PATH = "exp/lag_exp/bin/"
DEFAULT_CSV_PATH = "exp/lag_exp/csv/"


def get_existing_experiment_numbers(bin_path=DEFAULT_BIN_PATH):
    result_files = glob.glob(os.path.join(bin_path, "results*.npz"))
    matches = [
        re.search(".*results([0-9]*).*", fn, re.IGNORECASE) for fn in result_files
    ]
    experiment_numbers = [int(m.group(1)) for m in matches if m is not None]
    experiment_numbers.sort()
    return experiment_numbers


def gen_filename(n, bin_path=DEFAULT_BIN_PATH):
    return os.path.join(bin_path, "results%d.npz" % n)


def add_more_results(
    new_file_id,
    ms,
    seldonian_solutions_found,
    seldonian_fs,
    seldonian_failures_g1,
    seldonian_upper_bound,
    LS_solutions_found,
    LS_fs,
    LS_failures_g1,
    LS_upper_bound,
    bin_path=DEFAULT_BIN_PATH,
):
    new_file = np.load(gen_filename(new_file_id, bin_path))
    new_ms = new_file["ms"]

    new_seldonian_solutions_found = new_file["s_solutions_found"]
    new_seldonian_fs = new_file["s_fs"]
    new_seldonian_failures_g1 = new_file["s_failures_g1"]
    new_seldonian_upper_bound = new_file["s_upper_bound"]

    new_LS_solutions_found = new_file["LS_solutions_found"]
    new_LS_fs = new_file["LS_fs"]
    new_LS_failures_g1 = new_file["LS_failures_g1"]
    new_LS_upper_bound = new_file["LS_upper_bound"]

    if ms is None:
        return [
            new_ms,
            new_seldonian_solutions_found,
            new_seldonian_fs,
            new_seldonian_failures_g1,
            new_seldonian_upper_bound,
            new_LS_solutions_found,
            new_LS_fs,
            new_LS_failures_g1,
            new_LS_upper_bound,
        ]
    else:
        seldonian_solutions_found = np.vstack(
            [seldonian_solutions_found, new_seldonian_solutions_found]
        )
        seldonian_fs = np.vstack([seldonian_fs, new_seldonian_fs])
        seldonian_failures_g1 = np.vstack(
            [seldonian_failures_g1, new_seldonian_failures_g1]
        )
        seldonian_upper_bound = np.vstack(
            [seldonian_upper_bound, new_seldonian_upper_bound]
        )

        LS_solutions_found = np.vstack([LS_solutions_found, new_LS_solutions_found])
        LS_fs = np.vstack([LS_fs, new_LS_fs])
        LS_failures_g1 = np.vstack([LS_failures_g1, new_LS_failures_g1])
        LS_upper_bound = np.vstack([LS_upper_bound, new_LS_upper_bound])

        return [
            ms,
            seldonian_solutions_found,
            seldonian_fs,
            seldonian_failures_g1,
            seldonian_upper_bound,
            LS_solutions_found,
            LS_fs,
            LS_failures_g1,
            LS_upper_bound,
        ]


def stderror(v):
    non_nan = np.count_nonzero(~np.isnan(v))
    return np.nanstd(v, ddof=1) / np.sqrt(non_nan)


def save_to_csv(ms, results_qsa, results_ls, filename):
    n_cols = results_qsa.shape[1]

    with open(filename, mode="w") as file:
        writer = csv.writer(file, delimiter=",")

        for col in range(n_cols):
            cur_m = ms[col]
            seldonian_data = results_qsa[:, col]
            LS_data = results_ls[:, col]

            non_nan = np.count_nonzero(~np.isnan(seldonian_data))
            if non_nan > 0:
                seldonian_mean = np.nanmean(seldonian_data)
                seldonian_stderror = stderror(seldonian_data)
            else:
                seldonian_mean = "NaN"
                seldonian_stderror = "NaN"

            LS_mean = np.mean(LS_data)
            LS_stderror = stderror(LS_data)

            writer.writerow(
                [cur_m, seldonian_mean, seldonian_stderror, LS_mean, LS_stderror]
            )


def gather_results(bin_path=DEFAULT_BIN_PATH, csv_path=DEFAULT_CSV_PATH):
    ms = None
    seldonian_solutions_found = None
    seldonian_fs = None
    seldonian_failures_g1 = None
    seldonian_upper_bound = None
    LS_solutions_found = None
    LS_fs = None
    LS_failures_g1 = None
    LS_upper_bound = None

    experiment_numbers = get_existing_experiment_numbers(bin_path)

    for file_idx in experiment_numbers:
        res = add_more_results(
            file_idx,
            ms,
            seldonian_solutions_found,
            seldonian_fs,
            seldonian_failures_g1,
            seldonian_upper_bound,
            LS_solutions_found,
            LS_fs,
            LS_failures_g1,
            LS_upper_bound,
            bin_path,
        )
        [
            ms,
            seldonian_solutions_found,
            seldonian_fs,
            seldonian_failures_g1,
            seldonian_upper_bound,
            LS_solutions_found,
            LS_fs,
            LS_failures_g1,
            LS_upper_bound,
        ] = res

    if ms is None or seldonian_fs is None or LS_fs is None:
        logger.warning("No experiment results found.")
        return

    save_to_csv(ms, -seldonian_fs, -LS_fs, os.path.join(csv_path, "fs.csv"))
    save_to_csv(
        ms,
        seldonian_solutions_found,
        LS_solutions_found,
        os.path.join(csv_path, "solutions_found.csv"),
    )
    save_to_csv(
        ms,
        seldonian_failures_g1,
        LS_failures_g1,
        os.path.join(csv_path, "failures_g1.csv"),
    )
    save_to_csv(
        ms,
        seldonian_upper_bound,
        LS_upper_bound,
        os.path.join(csv_path, "upper_bound.csv"),
    )
