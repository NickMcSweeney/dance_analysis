from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import numpy as np
import os
import pandas as pd
import math
import re
import csv

from utils import *


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
    btm = math.sqrt(normX ** 2) * math.sqrt(normY ** 2)
    return top / btm

for index in range(13):
    # test each label in dataset for it's corelation to 
