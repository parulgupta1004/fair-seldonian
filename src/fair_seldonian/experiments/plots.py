import matplotlib.pyplot as plt
from .results import gather_results
import numpy as np

csv_path = 'exp/lag_exp/csv/'
img_path = 'exp/lag_exp/images/'


def loadAndPlotResults(fileName, ylabel, output_file, is_yAxis_prob, legend_loc):
    """
    This function is used to plot the results from the csv files and store the final graph

    :param filename: The csv file path from where the data is imported
    :param ylabel: The lable on the Y-axis of the graph
    :param output_file: The path where the graph image must be stored
    :param is_yAxis_prob: Bool of whether the Y-axis is probabity value or not
    :param legend_loc: The location of the legend
    """
    file_ms, file_QSA, file_QSA_stderror, file_LS, file_LS_stderror = np.loadtxt(fileName, delimiter = ',',
                                                                                   unpack = True )

    fig = plt.figure()

    plt.xlim(min(file_ms), max(file_ms))
    plt.xlabel( "Amount of data", fontsize = 16 )
    plt.xscale( 'log' )
    plt.xticks( fontsize = 12 )
    plt.ylabel( ylabel, fontsize = 16 )

    if is_yAxis_prob:
        plt.ylim(-0.1, 1.1)

    plt.plot ( file_ms, file_QSA, 'b-', linewidth = 3, label = 'QSA' )
    plt.errorbar ( file_ms, file_QSA, yerr = file_QSA_stderror, fmt = '.k' );
    plt.plot ( file_ms, file_LS, 'r-', linewidth = 3, label = 'LogRes' )
    plt.errorbar ( file_ms, file_LS, yerr = file_LS_stderror, fmt = '.k' );
    plt.legend ( loc = legend_loc, fontsize = 12 )
    plt.tight_layout ()

    plt.savefig( output_file, bbox_inches = 'tight' )
    plt.show( block = False )


if __name__ == "__main__":
    gather_results()

    loadAndPlotResults(csv_path + 'fs.csv', 'Log Loss', img_path + 'tutorial7MSE_py.png', False, 'lower right')
    loadAndPlotResults(csv_path + 'solutions_found.csv', 'Probability of Solution',
                         img_path + 'tutorial7PrSoln_py.png', True, 'best' )
    loadAndPlotResults(csv_path + 'failures_g1.csv', r'Probability of $g(a(D))>0$',
                         img_path + 'tutorial7PrFail1_py.png', True, 'best' )
    loadAndPlotResults(csv_path+'upper_bound.csv',     r'upper bound', img_path+'tutorial7PrFail2_py.png', False,  'best')
    plt.show()
