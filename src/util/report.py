import time
from util.eval import get_corrs, get_r2
from matplotlib import pyplot as plt
import numpy as np


def report_time(fn, desc):
    print(desc)
    start_t = time.time()
    ret = fn()
    end_t = time.time()
    print(f"Done in {end_t - start_t:.2f} seconds.")
    return ret


def report_metrics(values, true_values):
    pearson, spearman = get_corrs(values, true_values)
    r2 = get_r2(values, true_values)
    print("Correlations:")
    print(f"\tPearson: {np.average(pearson):.3f} ({np.median(pearson):.3f})")
    print(f"\tSpearman: {np.average(spearman):.3f} ({np.median(spearman):.3f})")
    print(f"\tR2: {r2[0]:.3f}")


def plot_metrics(values, true_values):
    pearson, spearman = get_corrs(values, true_values)
    r2 = get_r2(values, true_values)
    plt.boxplot([pearson, spearman, r2])
    plt.yscale("log")
    plt.show()