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

from sklearn.model_selection import KFold

from utils import *

mpl.rcParams["figure.figsize"] = (8, 6)
mpl.rcParams["axes.grid"] = False

csv_path = "kizomba2.csv"
valeria_path = "kizomba2_valeria.csv"
andrii_path = "kizomba2_andrii.csv"
labels_path = "kizomba2_labels.csv"
data_path = "kizomba2_formatted.csv"


SELECTED_FEATURES = read_in_selected_features(data_path)

past_history = 100
future_target = 1
STEP = 3
BATCH_SIZE = 10
BUFFER_SIZE = 10000
KFoldSplit = 5

pd.set_option("display.float_format", lambda x: "%.4f" % x)
df = pd.read_csv(data_path)

print(df.head())


tf.random.set_seed(13)
## NOTE: how do the time series labels get implemented
# using this lstm as a clasification method.
features = df[SELECTED_FEATURES[2:]]
features.index = df["timestamp"]
print(features.head())

labels = df["label"]
labels.index = df["timestamp"]
print(labels.head())

dataset = features.values

DATA_LEN = int(len(dataset))
DATA_STEP = int(DATA_LEN / 5)
print("training split: ", DATA_STEP)

data_mean = dataset.mean(axis=0)
data_std = dataset.std(axis=0)

dataset = (dataset - data_mean) / data_std
labelset = labels.values

dataarray = []
in_shape = None
for i in range(0, DATA_LEN, DATA_STEP):
    print("data ", i)

    x_single, y_single = multivariate_data(
        dataset,
        labelset,
        i,
        DATA_STEP + 1,
        past_history,
        future_target,
        STEP,
        single_step=True,
    )
    if in_shape == None:
        in_shape = x_single.shape[-2:]
    dataarray.append((x_single, y_single))

print("input shape ", in_shape)
single_step_model = tf.keras.models.Sequential()
single_step_model.add(tf.keras.layers.LSTM(156, input_shape=in_shape))
single_step_model.add(tf.keras.layers.Dense(13))

single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss="mae")

EPOCHS = 10
EVALUATION_INTERVAL = 20

for i in range(5):
    data_tupples = dataarray
    val_data_single_tupple = data_tupples.pop(i)
    x_train_single = []
    y_train_single = []
    for (x, y) in data_tupples:
        x_train_single = x_train_single + x
        y_train_single = y_train_single + y
    x_val_single = val_data_single_tupple[0]
    y_val_single = val_data_single_tupple[1]

    print("Single window of training data : {}".format(x_train_single[0].shape))
    print("Single window of validation data : {}".format(x_val_single[0].shape))
    print("y training example: {}".format(y_train_single[800]))

    train_data_single = tf.data.Dataset.from_tensor_slices(
        (x_train_single, y_train_single)
    )
    train_data_single = (
        train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    )

    val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
    val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

    # test train
    single_step_history = single_step_model.fit(
        train_data_single,
        epochs=EPOCHS,
        steps_per_epoch=EVALUATION_INTERVAL,
        validation_data=val_data_single,
        validation_steps=5,
    )

    plot_train_history(single_step_history, "Single Step Training and validation loss")


# for x, y in val_data_single.take(1):
# print(x.shape)
# print(y.shape)
# # print(x)
# print(y[-1])
