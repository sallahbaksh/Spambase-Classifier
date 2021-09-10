import numpy as np
import math


def confusion_matrix(y_pred, test_y):
    TP, TN, FP, FN = 0, 0, 0, 0
    for pred in range(0, len(y_pred)):
        if y_pred[pred] == 1 and test_y[pred] == 1:
            TP += 1
        elif y_pred[pred] == 0 and test_y[pred] == 0:
            TN += 1
        elif y_pred[pred] == 1 and test_y[pred] == 0:
            FP += 1
        elif y_pred[pred] == 0 and test_y[pred] == 1:
            FN += 1

    return TP, TN, FP, FN


def calc_accuracy(TP, TN, test_size):
    return (TP + TN) / test_size


def calc_precision(TP, FP):
    return TP / (TP + FP)


def calc_recall(TP, FN):
    return TP / (TP + FN)


def calc_f_measure(TP, FP, FN):
    return (2 * calc_precision(TP, FP) * calc_recall(TP, FN)) / (calc_precision(TP, FP) + calc_recall(TP, FN))
