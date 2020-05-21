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

data_mean = dataset.mean(axis=0)
data_std = dataset.std(axis=0)

dataset = (dataset - data_mean) / data_std
# labelset = labels.values
l = []
for val in labels.values:
    row = np.zeros(13)
    row[val] = 1
    l.append(row)
labelset = l

X, y = multivariate_data(
    dataset, labelset, 0, None, past_history, future_target, STEP, single_step=True,
)
kf = KFold(n_splits=5)
kf.get_n_splits(X)

model_data = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    print("Single window of training data : {}".format(X_train[0].shape))
    print("Single window of validation data : {}".format(X_test[0].shape))
    print("y training example: {}".format(y_train[800]))

    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_data = test_data.batch(BATCH_SIZE).repeat()

    model_data.append((train_data, test_data))

model = tf.keras.models.Sequential()
print("shape ", X.shape[-2:])
model.add(tf.keras.layers.LSTM(100, input_shape=X.shape[-2:]))
model.add(tf.keras.layers.Dense(13))
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss="mae")

EPOCHS = 10
EVALUATION_INTERVAL = 20

for (train_data, test_data) in model_data:
    # test train
    history = model.fit(
        train_data,
        epochs=EPOCHS,
        steps_per_epoch=EVALUATION_INTERVAL,
        validation_data=test_data,
        validation_steps=5,
    )

    plot_train_history(history, "Training and validation loss")


# for x, y in val_data_single.take(1):
# print(x.shape)
# print(y.shape)
# # print(x)
# print(y[-1])
