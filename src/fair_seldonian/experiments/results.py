import numpy as np
import csv
import glob
import re

bin_path = 'exp/lag_exp/bin/'
csv_path = 'exp/lag_exp/csv/'


def get_existing_experiment_numbers():
    result_files = glob.glob(bin_path + 'results*.npz' )
    experiment_numbers = [re.search('.*results([0-9]*).*', fn, re.IGNORECASE) for fn in result_files ]
    experiment_numbers = [int(i.group(1)) for i in experiment_numbers]
    experiment_numbers.sort()
    return experiment_numbers


def genFilename(n):
    return bin_path + 'results%d.npz' % n


def addMoreResults(newFileId, ms, seldonian_solutions_found, seldonian_fs, seldonian_failures_g1, seldonian_upper_bound,
                   LS_solutions_found, LS_fs, LS_failures_g1, LS_upper_bound):
    newFile = np.load(genFilename(newFileId))
    new_ms = newFile['ms']

    new_seldonian_solutions_found = newFile['s_solutions_found']
    new_seldonian_fs = newFile['s_fs']
    new_seldonian_failures_g1 = newFile['s_failures_g1']
    new_seldonian_upper_bound = newFile['s_upper_bound']

    new_LS_solutions_found = newFile['LS_solutions_found']
    new_LS_fs = newFile['LS_fs']
    new_LS_failures_g1 = newFile [ 'LS_failures_g1']
    new_LS_upper_bound = newFile['LS_upper_bound']

    if ms is None:
        return [new_ms, new_seldonian_solutions_found, new_seldonian_fs, new_seldonian_failures_g1,
                 new_seldonian_upper_bound,
                 new_LS_solutions_found, new_LS_fs, new_LS_failures_g1, new_LS_upper_bound]
    else:
        seldonian_solutions_found = np.vstack([seldonian_solutions_found, new_seldonian_solutions_found])
        seldonian_fs = np.vstack([seldonian_fs, new_seldonian_fs])
        seldonian_failures_g1 = np.vstack([seldonian_failures_g1, new_seldonian_failures_g1])
        seldonian_upper_bound = np.vstack([seldonian_upper_bound, new_seldonian_upper_bound])

        LS_solutions_found = np.vstack([LS_solutions_found, new_LS_solutions_found])
        LS_fs = np.vstack([LS_fs, new_LS_fs])
        LS_failures_g1 = np.vstack([LS_failures_g1, new_LS_failures_g1])
        LS_upper_bound = np.vstack([LS_upper_bound, new_LS_upper_bound])

        return [ms, seldonian_solutions_found, seldonian_fs, seldonian_failures_g1, seldonian_upper_bound,
                 LS_solutions_found, LS_fs, LS_failures_g1, LS_upper_bound]


def stderror(v):
    non_nan = np.count_nonzero(~np.isnan(v))
    return np.nanstd(v, ddof = 1) / np.sqrt(non_nan)


def saveToCSV(ms, resultsQSA, resultsLS, filename):
    nCols = resultsQSA.shape[1]

    with open( filename, mode = 'w' ) as file:
        writer = csv.writer( file, delimiter = ',' )

        for col in range( nCols ):

            cur_m = ms[ col ]
            seldonian_data = resultsQSA[ :, col ]
            LS_data = resultsLS[ :, col ]

            non_nan = np.count_nonzero( ~np.isnan ( seldonian_data ) )
            if non_nan > 0:
                seldonian_mean = np.nanmean( seldonian_data )
                seldonian_stderror = stderror( seldonian_data )
            else:
                seldonian_mean = 'NaN'
                seldonian_stderror = 'NaN'

            LS_mean = np.mean( LS_data )
            LS_stderror = stderror( LS_data )

            writer.writerow([cur_m, seldonian_mean, seldonian_stderror, LS_mean, LS_stderror ] )


def gather_results():
    ms = None
    seldonian_solutions_found = None
    seldonian_fs = None
    seldonian_failures_g1 = None
    seldonian_upper_bound = None
    LS_solutions_found = None
    LS_fs = None
    LS_failures_g1 = None
    LS_upper_bound = None

    experiment_numbers = get_existing_experiment_numbers()

    for file_idx in experiment_numbers:
        res = addMoreResults(file_idx,
                               ms,
                               seldonian_solutions_found, seldonian_fs, seldonian_failures_g1, seldonian_upper_bound,
                               LS_solutions_found, LS_fs, LS_failures_g1, LS_upper_bound)
        [ms, seldonian_solutions_found, seldonian_fs, seldonian_failures_g1, seldonian_upper_bound,
          LS_solutions_found, LS_fs, LS_failures_g1, LS_upper_bound] = res

    saveToCSV(ms, -seldonian_fs, -LS_fs, csv_path + 'fs.csv' )
    saveToCSV(ms, seldonian_solutions_found, LS_solutions_found, csv_path + 'solutions_found.csv' )
    saveToCSV(ms, seldonian_failures_g1, LS_failures_g1, csv_path + 'failures_g1.csv')
    saveToCSV(ms,  seldonian_upper_bound,      LS_upper_bound,     csv_path+'upper_bound.csv')
