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
data_path = "kizomba2_formatted.csv"


ALL_FEATURES = []
laban_elems = []
labels = []
LABAN_FEATURES = [
    "feet_hip",
    "hands_shoulder",
    "hands",
    "hands_head",
    "hands_hip",
    "hip_ground",
    "hip_ground_feet",
    "gait",
    "head_orientation_phi",
    "head_orientation_theta",
    "head_orientation_psi",
    "decel_peaks",
    "hip_vel",
    "hands_vel",
    "feet_vel",
    "hip_acc",
    "hands_acc",
    "feet_acc",
    "jerk",
    "vol_5",
    "vol_all",
    "vol_upper",
    "vol_lower",
    "vol_l",
    "vol_r",
    "torso_height",
    "hands_level",
    "total_dist",
    "total_area",
]

dance_len = 0

print("read in valeria features")
f1 = read_in_features(valeria_path)
print("read in valeria data")
l1 = read_in_dancer(valeria_path, f1)
dance_len = len(l1)
print("create valeria data frames")
l1 = featureFrames(l1, 175)

print("read in andrii features")
f2 = read_in_features(andrii_path)
print("read in andrii data")
l2 = read_in_dancer(andrii_path, f2)
print("create andrii data frames")
l2 = featureFrames(l2, 175)

print("read in labels")
labels = read_in_labels(labels_path, dance_len)

print("updating the label objects")
for index in range(len(l1)):
    laban_elems.append(labels[index])
    if index == 0:
        for feature in LABAN_FEATURES:
            if feature in [
                "decel_peaks",
                "hands_level",
                "total_dist",
                "total_area",
            ]:
                laban_elems[index].append("V_" + feature)
                laban_elems[index].append("A_" + feature)
            elif feature in ["hip_vel", "hands_vel", "feet_vel"]:
                laban_elems[index].append("V_" + feature + "_max")
                laban_elems[index].append("A_" + feature + "_max")
                laban_elems[index].append("V_" + feature + "_min")
                laban_elems[index].append("A_" + feature + "_min")
                laban_elems[index].append("V_" + feature + "_std")
                laban_elems[index].append("A_" + feature + "_std")
            elif feature in ["hip_acc", "hands_acc", "feet_acc", "jerk"]:
                laban_elems[index].append("V_" + feature + "_max")
                laban_elems[index].append("A_" + feature + "_max")
                laban_elems[index].append("V_" + feature + "_std")
                laban_elems[index].append("A_" + feature + "_std")
            else:
                laban_elems[index].append("V_" + feature + "_max")
                laban_elems[index].append("A_" + feature + "_max")
                laban_elems[index].append("V_" + feature + "_min")
                laban_elems[index].append("A_" + feature + "_min")
                laban_elems[index].append("V_" + feature + "_mean")
                laban_elems[index].append("A_" + feature + "_mean")
                laban_elems[index].append("V_" + feature + "_std")
                laban_elems[index].append("A_" + feature + "_std")

    elif index == 1:
        print("exising row --> ", laban_elems[index])
        for j in range(len(l1[index])):
            print("j is --> ", j)
            laban_elems[index].append(l1[index][j])
            print("l1 is --> ", l1[index][j])
            laban_elems[index].append(l2[index][j])
            print("l2 is --> ", l2[index][j])
    else:
        for j in range(len(l1[index])):
            laban_elems[index].append(l1[index][j])
            laban_elems[index].append(l2[index][j])

with open(data_path, "w") as out:
    writer = csv.writer(out)
    for index, elem in enumerate(laban_elems):
        row = labels[index] + elem
        writer.writerow(row)
