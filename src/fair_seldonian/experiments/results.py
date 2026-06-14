from __future__ import annotations

import csv
import glob
import logging
import os
import re

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_BIN_PATH = "exp/lag_exp/bin/"
DEFAULT_CSV_PATH = "exp/lag_exp/csv/"


def get_existing_experiment_numbers(bin_path: str = DEFAULT_BIN_PATH) -> list[int]:
    result_files = glob.glob(os.path.join(bin_path, "results*.npz"))
    matches = [
        re.search(".*results([0-9]*).*", fn, re.IGNORECASE) for fn in result_files
    ]
    experiment_numbers = [int(m.group(1)) for m in matches if m is not None]
    experiment_numbers.sort()
    return experiment_numbers


def gen_filename(n: int, bin_path: str = DEFAULT_BIN_PATH) -> str:
    return os.path.join(bin_path, f"results{n}.npz")


def add_more_results(
    new_file_id: int,
    ms: np.ndarray | None,
    seldonian_solutions_found: np.ndarray | None,
    seldonian_fs: np.ndarray | None,
    seldonian_failures_g1: np.ndarray | None,
    seldonian_upper_bound: np.ndarray | None,
    LS_solutions_found: np.ndarray | None,
    LS_fs: np.ndarray | None,
    LS_failures_g1: np.ndarray | None,
    LS_upper_bound: np.ndarray | None,
    bin_path: str = DEFAULT_BIN_PATH,
) -> list[np.ndarray]:
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
        # Once ms is populated the accumulators are too (set together below).
        assert (
            seldonian_solutions_found is not None
            and seldonian_fs is not None
            and seldonian_failures_g1 is not None
            and seldonian_upper_bound is not None
            and LS_solutions_found is not None
            and LS_fs is not None
            and LS_failures_g1 is not None
            and LS_upper_bound is not None
        )
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


def stderror(v: np.ndarray) -> float:
    non_nan = np.count_nonzero(~np.isnan(v))
    if non_nan < 2:
        return float("nan")
    return float(np.nanstd(v, ddof=1) / np.sqrt(non_nan))


def save_to_csv(
    ms: np.ndarray, results_qsa: np.ndarray, results_ls: np.ndarray, filename: str
) -> None:
    n_cols = results_qsa.shape[1]

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    with open(filename, mode="w", newline="") as file:
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


def gather_results(
    bin_path: str = DEFAULT_BIN_PATH, csv_path: str = DEFAULT_CSV_PATH
) -> None:
    ms: np.ndarray | None = None
    seldonian_solutions_found: np.ndarray | None = None
    seldonian_fs: np.ndarray | None = None
    seldonian_failures_g1: np.ndarray | None = None
    seldonian_upper_bound: np.ndarray | None = None
    LS_solutions_found: np.ndarray | None = None
    LS_fs: np.ndarray | None = None
    LS_failures_g1: np.ndarray | None = None
    LS_upper_bound: np.ndarray | None = None

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

    # If results were gathered, every accumulator was populated together.
    assert (
        seldonian_solutions_found is not None
        and seldonian_failures_g1 is not None
        and seldonian_upper_bound is not None
        and LS_solutions_found is not None
        and LS_failures_g1 is not None
        and LS_upper_bound is not None
    )

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
