from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import numpy as np
import os
import pandas as pd
import math
import re
import csv

from utils import *

data_path = "kizomba2_formatted.csv"
# data_path = "kizomba2_normalized.csv"


def pccPop(datX, datY):
    ## calculate the pearson correlation coefficient for a population
    datX = np.array(datX)
    datY = np.array(datY)
    rhoXY = np.cov(datX, datY) / (datX.std() * datY.std())
    return rhoXY


def pccSample(sampleX, sampleY):
    X = np.array(sampleX)
    Y = np.array(sampleY)
    normX = X - X.mean()
    normY = Y - Y.mean()
    top = (normX * normY).sum()
    btm = math.sqrt((np.square(normX)).sum()) * math.sqrt((np.square(normY)).sum())
    print(top)
    print(btm)
    return top / btm


def filterDataOnLabel(data, label_id):
    out = []
    for row in data:
        if row[0] == label_id:
            out.append(row)
    return out


# main fn
if __name__ == "__main__":
    ###
    # Pre process dance files
    # builds a csv file with the correct formatting of features
    ###
    SELECTED_FEATURES = read_in_selected_features(data_path)

    pd.set_option("display.float_format", lambda x: "%.4f" % x)
    df = pd.read_csv(data_path)

    print(df.head())

    features = df[SELECTED_FEATURES[1:]]
    features.index = df["timestamp"]

    labels = df["label"]
    labels.index = df["timestamp"]

    # compare first 3 labels
    l1 = filterDataOnLabel(features.values, 2)
    l2 = filterDataOnLabel(features.values, 5)
    l3 = filterDataOnLabel(features.values, 4)
    lens = [len(l1), len(l2), len(l3)]
    print(lens)
    minLen = int(min(lens) / 2)
    l1a = np.array(l1[0:1000])[:, 1:]
    l1b = np.array(l1[1000:2000])[:, 1:]
    print(len(l1a), " .. ", len(l1b))
    l2a = np.array(l2[0:1000])[:, 1:]
    l2b = np.array(l2[1000:2000])[:, 1:]
    print(len(l2a), " .. ", len(l2b))
    l3a = np.array(l3[0:1000])[:, 1:]
    l3b = np.array(l3[1000:2000])[:, 1:]
    print(len(l3))

    # pcc_l11 = pccPop(l1, l1)
    # pcc_l12 = pccPop(l1, l2)
    # pcc_l13 = pccPop(l1, l2)

    # print("vals: ", pcc_l11)
    # print("col ", len(pcc_l12[0]))
    # print("row ", len(pcc_l12))
    # res = []
    # for i in range(len(pcc_l12)):
    # for j in range(len(pcc_l12[i])):
    # if i == j:
    # res.append(pcc_l12[i][j])

    # print("vals: ", res)
    # print("vals: ", pcc_l13)

    corrs = []
    for i in range(len(l1a[0])):
        corr, _ = pearsonr(l1a[:, i], l1b[:, i])
        corrs.append(abs(corr))
    corrs = np.array(corrs)
    l1index = np.argwhere(corrs > 0.4)
    # print(l1index)
    corrs = corrs[~np.isnan(corrs)]
    print("Pearsons correlation: %.3f" % corrs.mean())

    corrs = []
    for i in range(len(l2a[0])):
        corr, _ = pearsonr(l2a[:, i], l2b[:, i])
        corrs.append(abs(corr))
    corrs = np.array(corrs)
    l2index = np.argwhere(corrs > 0.4)
    # print(l2index)
    corrs = corrs[~np.isnan(corrs)]
    print("Pearsons correlation: %.3f" % corrs.mean())

    corrs = []
    for i in range(len(l3a[0])):
        corr, _ = pearsonr(l3a[:, i], l3b[:, i])
        corrs.append(abs(corr))
    corrs = np.array(corrs)
    l3index = np.argwhere(corrs > 0.4)
    # print(l2index)
    corrs = corrs[~np.isnan(corrs)]
    print("Pearsons correlation: %.3f" % corrs.mean())

    l1 = np.concatenate((l1a, l1b), axis=0)
    l2 = np.concatenate((l2a, l2b), axis=0)
    l3 = np.concatenate((l3a, l3b), axis=0)
    corrs = []
    for i in range(len(l1[0])):
        corr, _ = pearsonr(l1[:, i], l2[:, i])
        corrs.append(abs(corr))
    corrs = np.array(corrs)
    lindex_corr1 = np.argwhere(corrs > 0.4)
    lindex = np.argwhere(corrs < 0.5)
    corrs = []
    for i in range(len(l1[0])):
        corr, _ = pearsonr(l1[:, i], l3[:, i])
        corrs.append(abs(corr))
    corrs = np.array(corrs)
    lindex_corr2 = np.argwhere(corrs > 0.4)
    corrs = corrs[~np.isnan(corrs)]
    print("Pearsons correlation: %.3f" % corrs.mean())

    valid_features = []
    for i in range(len(l1[0])):
        if (i in l1index or i in l2index or i in l3index) and (
            i not in lindex_corr1 and i not in lindex_corr2
        ):
            valid_features.append(i)
    print("valid features [", len(valid_features), "]:\n", valid_features)

    l1a = l1a[:, valid_features]
    l1b = l1b[:, valid_features]
    l2a = l2a[:, valid_features]
    l2b = l2b[:, valid_features]
    l3a = l3a[:, valid_features]
    l3b = l3b[:, valid_features]

    corrs = []
    for i in range(len(l1a[0])):
        corr, _ = pearsonr(l1a[:, i], l1b[:, i])
        corrs.append(abs(corr))
    corrs = np.array(corrs)
    # print(l1index)
    corrs = corrs[~np.isnan(corrs)]
    print("Pearsons correlation: %.3f" % corrs.mean())

    corrs = []
    for i in range(len(l2a[0])):
        corr, _ = pearsonr(l2a[:, i], l2b[:, i])
        corrs.append(abs(corr))
    corrs = np.array(corrs)
    # print(l2index)
    corrs = corrs[~np.isnan(corrs)]
    print("Pearsons correlation: %.3f" % corrs.mean())

    corrs = []
    for i in range(len(l3a[0])):
        corr, _ = pearsonr(l3a[:, i], l3b[:, i])
        corrs.append(abs(corr))
    corrs = np.array(corrs)
    # print(l2index)
    corrs = corrs[~np.isnan(corrs)]
    print("Pearsons correlation: %.3f" % corrs.mean())

    l1 = np.concatenate((l1a, l1b), axis=0)
    l2 = np.concatenate((l2a, l2b), axis=0)
    l3 = np.concatenate((l3a, l3b), axis=0)
    print("lens: ", len(l1), " // ", len(l2), " // ", len(l3))
    corrs = []
    for i in range(len(l1[0])):
        corr, _ = pearsonr(l1[:, i], l2[:, i])
        corrs.append(abs(corr))
    corrs = np.array(corrs)
    corrs = corrs[~np.isnan(corrs)]
    print("Pearsons correlation l1&l2: %.3f" % corrs.mean())
    corrs = []
    for i in range(len(l1[0])):
        corr, _ = pearsonr(l3[:, i], l2[:, i])
        corrs.append(abs(corr))
    corrs = np.array(corrs)
    corrs = corrs[~np.isnan(corrs)]
    print("Pearsons correlation l1&l3: %.3f" % corrs.mean())
    corrs = []
    for i in range(len(l1[0])):
        corr, _ = pearsonr(l1[:, i], l3[:, i])
        corrs.append(abs(corr))
    corrs = np.array(corrs)
    corrs = corrs[~np.isnan(corrs)]
    print("Pearsons correlation l2&l3: %.3f" % corrs.mean())

    # corrs = []
    # for i in range(len(l1[0])):
    # corr, _ = spearmanr(l1a[:, i], l1b[:, i])
    # corrs.append(abs(corr))
    # corrs = np.array(corrs)
    # print(corrs)
    # corrs = corrs[~np.isnan(corrs)]
    # print("Spearmans correlation: %.3f" % corrs.mean())
    # corr, _ = spearmanr(l1, l2)
    # print("Spearmans correlation: %.3f" % corr)
