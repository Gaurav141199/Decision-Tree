import numpy as np
import pandas as pd


def read_csv(t_file):
    x_t = pd.read_csv(t_file)
    x_t = np.array(x_t.iloc[:, :])
    m, n = x_t.shape
    y_t = np.array([x[n - 1] for x in x_t], dtype=int)
    x_t = np.array([x[:n - 1] for x in x_t], dtype=int)
    return x_t, y_t


def median_cont(x_t):
    median = [np.median(x_t[:, i]) for i in range(10)]
    x_new = x_t.copy()
    for x in x_new:
        for i in range(0, 10):
            if x[i] >= median[i]:
                x[i] = 1
            else:
                x[i] = 0
    x_new = np.array(x_new)
    return x_new


def split_data(x_t, y_t, column):
    x_left = []
    x_right = []
    y_left = []
    y_right = []
    for i in range(len(x_t)):
        if x_t[i][column] == 0:
            x_left.append(x_t[i])
            y_left.append(y_t[i])
        else:
            x_right.append(x_t[i])
            y_right.append(y_t[i])
    return np.array(x_left), np.array(x_right), np.array(y_left), np.array(y_right)


def entropy(y_t):
    uniques = np.unique(y_t, return_counts=True)
    classes = uniques[0]
    counts = uniques[1]
    array = [(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(classes))]
    return np.sum(array)


def gain_info(x_t, y_t, column):
    entropy_total = entropy(y_t)
    x_t_left, x_t_right, y_t_left, y_t_right = split_data(x_t, y_t, column)
    left_count = len(y_t_left)
    right_count = len(y_t_right)
    entropy_left = entropy(y_t_left)
    entropy_right = entropy(y_t_right)
    entropy_weight = (left_count * entropy_left + right_count * entropy_right)/(left_count+right_count)
    return entropy_total - entropy_weight


def best_feature(x_t, y_t):
    column_best = 0
    best_info = -np.inf
    for i in range(0, len(x_t[0])):
        info_gain = gain_info(x_t, y_t, i)
        if best_info < info_gain:
            column_best = i
            best_info = info_gain
    return column_best