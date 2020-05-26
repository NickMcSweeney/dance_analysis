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
RELATIONAL_FEATURES = [
    ## pressure
    "hands_vel_A",  # velocity of hands side A [max/min/mean/std] - 0
    "hands_vel_B",  # velocity of hands side B [max/min/mean/std] - 1
    "hands_force_A",  # force of hands side A [max/min/mean/std] - 2
    "hands_force_B",  # force of hands side B [max/min/mean/std] - 3
    "hip_dist",  # distance between hips [max/min/mean/std] - 4
    "shoulder_vel_diff",  # difference in shoulder velocities [mean/std] - 5
    "shoulder_dist",  # distance between partners shoulders [max/min/mean/std]- 6
    ## weight
    "shoulder_hips_A",  # xy dist between hips and shouler midpoint, person A [mean/std] - 7
    "shoulder_hips_B",  # xy dist between hips and shouler midpoint, person B [mean/std] - 8
    "hips_feet_A",  # xy plane dist between hips and feet midpoint, person A [mean/std] - 9
    "hips_feet_B",  # xy plane dist between hips and feet midpoint, person B [mean/std] - 10
    "hips_vel_diff",  # difference in velocity of hips [mean/std] - 11
    "hip_acc_diff",  # difference in acceleration of hips [mean/std] - 12
    "com_A",  # center of mass of person A [mean/std] - 13
    "com_B",  # center of mass of person B [mean/std] - 14
    "com_comb",  # combigned center of mass [mean/std] - 15
    ## possition
    "gait_A",  # distance between feet person A [max/min/mean/std] - 16
    "gait_B",  # distance between feet person B [max/min/mean/std] - 17
    "feet_vel",  # velocity of feet [mean/std] - 18
    "hip_orientation_diff",  # difference in orientation of hips (yaw) [mean/std] - 19
    "hand_dist_A",  # side A distance between partners hands [max/min/mean/std] - 20
    "hand_dist_B",  # side B distance between partners hands [max/min/mean/std] - 21
    "volume",  # combigned volume of dancers [max/min/mean/std] - 22
    "feet_volume",  # space feet take up [max/min/mean/std] - 23
]


def ReadInCouple(personOnePath, personTwoPath):
    SKELETON_LABELS = read_in_features(personOnePath)

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

    SKELETON_LABELS.append("Feet_X")
    SKELETON_LABELS.append("Feet_Y")
    SKELETON_LABELS.append("Feet_Z")
    SKELETON_LABELS.append("Shoulder_X")
    SKELETON_LABELS.append("Shoulder_Y")
    SKELETON_LABELS.append("Shoulder_Z")

    feetIndex = SKELETON_LABELS.index("Feet_X")
    shoulderIndex = SKELETON_LABELS.index("Shoulder_X")

    movementA = {"hips": 0, "lhand": 0, "rhand": 0, "feet": 0, "shoulders": 0}
    movementB = {"hips": 0, "lhand": 0, "rhand": 0, "feet": 0, "shoulders": 0}

    features = []
    for i in range(len(lead)):
        frame = []
        frame_lead = lead_n[i]
        frame_follow = follow_n[i]
        j = 0
        rowA = []
        rowB = []
        for elemA, elemB in zip(frame_lead, frame_follow):
            if j != 0 and j < 4:
                rowA.append(float(elemA))
                rowB.append(float(elemB))
            elif j > 3:
                rowA.append(None)
                rowB.append(None)
            j = (j + 1) % 8
        frame_lead = lead[i]
        frame_follow = follow[i]
        j = 0
        index = 0
        for elemA, elemB in zip(frame_lead, frame_follow):
            if j > 3:
                if len(rowA) < index:
                    break
                rowA[index] = float(elemA)
                rowB[index] = float(elemB)
            if j != 0:
                index = index + 1
            j = (j + 1) % 8

        handsLA = rowA[64:67]
        handsRA = rowA[92:95]
        handsLB = rowB[64:67]
        handsRB = rowB[92:95]
        feetLA = rowA[113:116]
        feetRA = rowA[141:144]
        feetA = [(x + y) / 2 for x, y in zip(feetLA, feetRA)]
        feetLB = rowB[113:116]
        feetRB = rowB[141:144]
        feetB = [(x + y) / 2 for x, y in zip(feetLB, feetRB)]
        shoulderLA = rowA[43:46]
        shoulderRA = rowA[71:74]
        shouldersA = [(x + y) / 2 for x, y in zip(shoulderLA, shoulderRA)]
        shoulderLB = rowB[43:46]
        shoulderRB = rowB[71:74]
        shouldersB = [(x + y) / 2 for x, y in zip(shoulderLB, shoulderRB)]
        hipsA = rowA[1:4]
        hipsB = rowB[1:4]
        hipQA = rowA[4:8]
        hipQB = rowB[4:8]

        rowA = rowA + feetA + shouldersA
        rowB = rowB + feetB + shouldersB

        movementA["hips"] = update_movement(rowA, movementA["hips"], 1)
        movementA["lhand"] = update_movement(rowA, movementA["lhand"], 64)
        movementA["rhand"] = update_movement(rowA, movementA["rhand"], 92)
        movementA["feet"] = update_movement(rowA, movementA["feet"], feetIndex)
        movementA["shoulders"] = update_movement(
            rowA, movementA["shoulders"], shoulderIndex
        )

        movementB["hips"] = update_movement(rowB, movementB["hips"], 1)
        movementB["lhand"] = update_movement(rowB, movementB["lhand"], 64)
        movementB["rhand"] = update_movement(rowB, movementB["rhand"], 92)
        movementB["feet"] = update_movement(rowB, movementB["feet"], feetIndex)
        movementB["shoulders"] = update_movement(
            rowB, movementB["shoulders"], shoulderIndex
        )

        ## hands velocity
        frame.append(
            (
                magnatude(movementA["lhand"].get_v())
                + magnatude(movementB["rhand"].get_v())
            )
            / 2
        )
        frame.append(
            (
                magnatude(movementA["rhand"].get_v())
                + magnatude(movementB["lhand"].get_v())
            )
            / 2
        )

        ## hands force
        frame.append(
            (
                magnatude(movementA["lhand"].get_f())
                + magnatude(movementB["rhand"].get_f())
            )
            / 2
        )
        frame.append(
            (
                magnatude(movementA["rhand"].get_f())
                + magnatude(movementB["lhand"].get_f())
            )
            / 2
        )

        ## hip distance
        frame.append(distance(hipsA, hipsB))

        ## shoulder velocity difference
        frame.append(
            magnatude(movementA["shoulders"].get_v())
            - magnatude(movementB["shoulders"].get_v())
        )

        ## distance between shoulders
        frame.append(distance(shouldersA, shouldersB))

        ## shoulder to hip offset
        frame.append(distance(shouldersA[:2] + [0], hipsA[:2] + [0]))
        frame.append(distance(shouldersB[:2] + [0], hipsB[:2] + [0]))

        ## hip to feet offset
        frame.append(distance(feetA[:2] + [0], hipsA[:2] + [0]))
        frame.append(distance(feetB[:2] + [0], hipsB[:2] + [0]))

        ## difference in hip velocities
        frame.append(
            magnatude(movementA["hips"].get_v()) - magnatude(movementB["hips"].get_v())
        )

        ## difference in hip acceleration
        frame.append(
            magnatude(movementA["hips"].get_a()) - magnatude(movementB["hips"].get_a())
        )

        ### center of mass
        comA = [0, 0, 0]
        comB = [0, 0, 0]
        totA = 0
        totB = 0

        for elemA, elemB in zip(rowA, rowB):
            if j != 0 and j < 4:
                r = re.compile(".*_X")
                comA[(j - 1) % 3] = comA[(j - 1) % 3] + float(elemA)
                comB[(j - 1) % 3] = comB[(j - 1) % 3] + float(elemB)
                totA = totA + float(elemA)
                totB = totB + float(elemB)
            j = (j + 1) % 7

        ## distance of com person A to hips person A
        frame.append(distance(hipsA, [comA[0] / totA, comA[1] / totA, comA[2] / totA]))

        ## distance of com person B to hips person B
        frame.append(distance(hipsB, [comB[0] / totB, comB[1] / totB, comB[2] / totB]))

        ## dancers center of mass distance
        frame.append(
            distance(
                [comA[0] / totA, comA[1] / totA, comA[2] / totA],
                [comB[0] / totB, comB[1] / totB, comB[2] / totB],
            )
        )

        ## gait
        frame.append(distance(feetLA, feetRA))
        frame.append(distance(feetLB, feetRB))

        ## feet velocity
        frame.append(
            (
                magnatude(movementA["feet"].get_v())
                + magnatude(movementB["feet"].get_v())
            )
            / 2
        )

        ## hip orientation
        orientation_diff = quat_difference(hipQA, hipQB)
        euler = quat_euler(orientation_diff)
        frame.append(euler[2])

        ## hands distance
        frame.append(distance(handsLA, handsRB))
        frame.append(distance(handsRA, handsLB))

        ## dancer space volume
        r = re.compile(".*_X")
        fullBody = list(filter(r.match, SKELETON_LABELS))
        bodyArray = []
        for jointName in fullBody:
            jointID = SKELETON_LABELS.index(jointName)
            bodyArray.append(rowA[jointID : jointID + 3])
            bodyArray.append(rowB[jointID : jointID + 3])

        frame.append(bboxVolume(bodyArray))

        ## volume taken by the feet
        frame.append(bboxVolume([feetLA, feetLB, feetRA, feetRB]))

        features.append(frame)
    return features


def FeatureCollection(rel_features, frame_size):
    lframe = []
    for index in range(len(rel_features) - frame_size):
        fframe = []
        rframe = np.array(rel_features[index : (index + frame_size)])
        for j in range(len(rel_features[index])):
            frame = rframe[:, j]
            if j in [5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19]:  # only mean andstd
                # MEAN
                fframe.append(frame.mean())
                # STD
                fframe.append(frame.std())
            else:  # all
                # MAX
                fframe.append(frame.max())
                # MIN
                fframe.append(frame.min())
                # MEAN
                fframe.append(frame.mean())
                # STD
                fframe.append(frame.std())
        lframe.append(fframe)
    return lframe


def formatHeaders(headers):
    formatted_headers = []
    for header in headers:
        if header in [
            "shoulder_vel_diff",
            "shoulder_hips_A",
            "shoulder_hips_B",
            "hips_feet_A",
            "hips_feet_B",
            "hips_vel_diff",
            "hip_acc_diff",
            "com_A",
            "com_B",
            "com_comb",
            "feet_vel",
            "hip_orientation_diff",
        ]:
            formatted_headers.append(header + "_mean")
            formatted_headers.append(header + "_std")
        else:
            formatted_headers.append(header + "_max")
            formatted_headers.append(header + "_min")
            formatted_headers.append(header + "_mean")
            formatted_headers.append(header + "_std")
    return formatted_headers


# main fn
if __name__ == "__main__":
    ###
    # Pre process dance files
    # builds a csv file with the correct formatting of features
    ###

    labels = []

    dance_len = 0

    features = ReadInCouple(andrii_path, valeria_path)
    features = FeatureCollection(features, 160)

    dance_len = len(features)
    print("read in labels (length ", dance_len, ")")
    labels = read_in_labels(labels_path, dance_len)

    features = np.array(features)
    print(features)

    RELATIONAL_FEATURES = formatHeaders(RELATIONAL_FEATURES)

    with open(data_path, "w") as out:
        writer = csv.writer(out)
        writer.writerow(labels[0] + RELATIONAL_FEATURES)
        for label, data in zip(labels[1:], features):
            if len(label) == 0 or len(data) == 0:
                break
            line = np.concatenate((label, data))
            writer.writerow(line)
