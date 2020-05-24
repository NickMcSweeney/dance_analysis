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
## Laban Elements
# Body
# Effort
# Shape
# Space
## Dance Elements
# Pressure
# Weight
# Possition
LABAN_FEATURES = [
    "hands_dist",  # average distance between hands
    "hands_vel",  # average velocity of hands
    "hands_force",  # average force of hands
    "gait",  # distance between feet
    "feet_vel",  # average velocity of feet
    "spine_hips",  # xy plane dist between hips and top of spine
    "hips_feet",  # xy plane dist between hips and feet
    "hips_feet_vel",  # difference in velocity between feet and hips
    "feet_dist",  # average dist between feet
    "hip_vel_diff",  # difference in hip velocities
    "hip_dist",  # distance between hips
    "shoulder_vel_diff",  # difference in shoulder velocities
    "shoulder_dist",  # average distance between shoulder
    "volume",  # combigned volume of dancers
]


def ReadInCouple(personOnePath, personTwoPath, features):
    pd.set_option("display.float_format", lambda x: "%.4f" % x)
    pOne = pd.read_csv(personOnePath)
    pTwo = pd.read_csv(personTwoPath)

    lead = pOne.values
    follow = pTwo.values

    for i in len(lead):
        frame_lead = lead[i]
        frame_follow = follow[i]

        ## Hands distance

        ## hands velocity

        ## hands force

        ## gait

        ## feet velocity

        ## spine to hip offset

        ## hip to feet offset

        ## hip velocity vs feet velocity

        ## feet distance

        ## difference in hip velocities

        ## hip distance

        ## difference in shoulder velocities

        ## distance between shoulders

        ## dancer space volume

    index = 0
    dancer_data = []
    movement = {"root": 0, "lhand": 0, "rhand": 0, "lfoot": 0, "rfoot": 0}
    with open(path_in, "r") as inp:
        timestamp = 0
        for row in csv.reader(inp):
            if index > 300000:
                break
            elif index != 0:
                i = 0
                exprow = [timestamp]
                for elem in row:
                    if i != 0:
                        exprow.append(float(elem))
                    i = (i + 1) % 8
                # root_elem = (exprow[0], exprow[1], exprow[2])
                # i = 0
                # temp = []
                # for elem in exprow:
                # if i < 3:
                # temp.append(elem - root_elem[i])
                # else:
                # temp.append(elem)
                # i = (i + 1) % 7
                # exprow = temp

                elem_id = FEATURES.index("Hips_X") + 1
                movement["root"] = update_movement(exprow, movement["root"], elem_id)
                elem_id = FEATURES.index("LeftHand_X") + 1
                movement["lhand"] = update_movement(exprow, movement["lhand"], elem_id)
                elem_id = FEATURES.index("RightHand_X") + 1
                movement["rhand"] = update_movement(exprow, movement["rhand"], elem_id)
                elem_id = FEATURES.index("LeftFoot_X") + 1
                movement["lfoot"] = update_movement(exprow, movement["lfoot"], elem_id)
                elem_id = FEATURES.index("RightFoot_X") + 1
                movement["rfoot"] = update_movement(exprow, movement["rfoot"], elem_id)
                features = laban_feature_pop(exprow, movement, FEATURES)
                dancer_data.append(features)
            index = index + 1
    return dancer_data


# main fn
if __name__ == "__main__":
    ###
    # Pre process dance files
    # builds a csv file with the correct formatting of features
    ###

    ALL_FEATURES = []
    laban_elems = []
    labels = []

    dance_len = 0

    print("read in valeria features")
    f1 = read_in_features(valeria_path)
    print("read in andrii features")
    f2 = read_in_features(andrii_path)

    print("read in valeria data")
    l1 = read_in_dancer(valeria_path, f1)
    print("read in andrii data")
    l2 = read_in_dancer(andrii_path, f2)

    dance_len = len(l1)
    print("read in labels (length ", dance_len, ")")
    labels = read_in_labels(labels_path, dance_len)

    print("create valeria data frames")
    l1 = featureFrames(l1, 160)[0 : len(labels)]
    print("create andrii data frames")
    l2 = featureFrames(l2, 160)[0 : len(labels)]

    print("updating the label objects")
    num_cols = len(l1[0])
    num_rows = len(l1) - 1
    print("the label length is: ", num_cols, " // ", num_rows)
    for index in range(num_rows):
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

        else:
            for j in range(num_cols):
                laban_elems[index].append(l1[index][j])
                laban_elems[index].append(l2[index][j])

    with open(data_path, "w") as out:
        writer = csv.writer(out)
        print("row length: ", len(laban_elems[0]), " // ", len(laban_elems[1]))
        for row in laban_elems:
            writer.writerow(row)
