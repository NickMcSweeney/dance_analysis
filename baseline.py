from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import math
import re
import csv

from utils import *

mpl.rcParams["figure.figsize"] = (8, 6)
mpl.rcParams["axes.grid"] = False

csv_path = "kizomba2.csv"
valeria_path = "kizomba2_valeria.csv"
andrii_path = "kizomba2_andrii.csv"
labels_path = "kizomba2_labels.csv"
data_path = "kizomba2_normalized.csv"


def ReadInLabels(path):
    skeleton_labels = []
    with open(path, "r") as inp:
        reader = csv.reader(inp)
        header = next(reader)
        index = 0
        for elem in header:
            if index == 0:
                prefix = elem
            elif index:
                skeleton_labels.append(prefix + "_" + elem)
                skeleton_labels.append(prefix + "_" + elem)
            index = (index + 1) % 8
        return skeleton_labels


def ReadInCouple(personOnePath, personTwoPath, SKELETON_LABELS):

    pd.set_option("display.float_format", lambda x: "%.4f" % x)
    pOne = pd.read_csv(personOnePath)
    pTwo = pd.read_csv(personTwoPath)

    lead = pOne.values
    lmean = lead.mean(axis=0)
    lstd = lead.std(axis=0)
    lead_n = (lead - lmean) / lstd

    follow = pTwo.values
    fmean = follow.mean(axis=0)
    fstd = follow.std(axis=0)
    follow_n = (follow - fmean) / fstd

    features = []
    for i in range(len(lead)):
        frame = []
        frame_lead = lead_n[i]
        frame_follow = follow_n[i]
        j = 0
        for elemA, elemB in zip(frame_lead, frame_follow):
            if j != 0 and j < 4:
                frame.append(float(elemA))
                frame.append(float(elemB))
            elif j > 3:
                frame.append(0)
                frame.append(0)
            j = (j + 1) % 8
        frame_lead = lead[i]
        frame_follow = follow[i]
        j = 0
        index = 0
        for elemA, elemB in zip(frame_lead, frame_follow):
            if j != 0 and j < 4:
                index = index + 2
            elif j > 3:
                frame[index] = float(elemA)
                frame[index + 1] = float(elemB)
                index = index + 2
            j = (j + 1) % 8

        features.append(frame)
    return features


# main fn
if __name__ == "__main__":
    ###
    # Pre process dance files
    # builds a csv file with the correct formatting of features
    ###

    labels = []

    dance_len = 0

    SKELETON_LABELS = ReadInLabels(andrii_path)

    features = ReadInCouple(andrii_path, valeria_path, SKELETON_LABELS)

    dance_len = len(features)
    print("read in labels (length ", dance_len, ")")
    labels = read_in_labels(labels_path, dance_len)

    features = np.array(features)
    print(features)

    with open(data_path, "w") as out:
        writer = csv.writer(out)
        writer.writerow(labels[0] + SKELETON_LABELS)
        for label, data in zip(labels[1:], features):
            if len(label) == 0 or len(data) == 0:
                break
            line = np.concatenate((label, data))
            writer.writerow(line)
