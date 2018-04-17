#!/usr/local/bin/python3

import csv
import numpy as np
import pickle
from collections import Counter
import math

def sample_handling():

    featureset = []

    with open('data.csv', 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        current_high = 0.0 #holds the current high of the row/day
        current_low = 0.0 #holds the current low of the row/day
        current_open = 0.0 #holds the current open of the row/day
        current_close = 0.0 #holds the current close of the row/day
        current_net_change = current_close - current_open #difference between current_open and current_close (this is the AVERAGE change through the day)

        prev_high = 0.0 #holds the previous row's/day's high value
        prev_low = 0.0 #holds the previous row's/day's low value
        prev_open = 0.0 #holds the previous row's/day's open value
        prev_close = 0.0 #holds the previous row's/day's close value
        prev_delta_high = 0.0 #holds the differences of the previous high and the high before that (in the nth row/day, it holds the difference of (n-1) and (n-2))
        prev_net_change = 0.0

        delta_high = 0.0 #holds change in high value between current row and previous row
        delta_low = 0.0 #holds change in low value between current row and previous row
        delta_open = 0.0 #holds change in open value between current row and previous row
        delta_close = 0.0 #holds change in close value between current row and previous row

        #counter variable
        counter = 0

        indicator = 0 #

        rows = list(reader)[-100000:] #rows is the last 100000 elements of reader, and thus the last 100000 rows of the csv file (most up to date data we have)
        #rows = list(reader)[-150000:] #last 150000 elements of reader
        for row in rows:

            current_high = float(row[2])
            current_low = float(row[3])
            current_open = float(row[1])
            current_close = float(row[4])

            if counter > 0:
                delta_high = current_high - prev_high
                delta_low = current_low - prev_low
                delta_open = current_open - prev_open
                delta_close = current_close - prev_close
                current_net_change = current_close - current_open

                if current_net_change > 0:
                    indicator = 1
                else:
                    indicator = -1

                featureset.append([[prev_delta_high, prev_high, indicator], [current_high]])

            prev_high = current_high
            prev_low = current_low
            prev_open = current_open
            prev_close = current_close
            prev_delta_high = delta_high
            prev_net_change = current_net_change

            counter += 1

        return featureset

def create_feature_sets_and_labels(test_size=0.1):
    features = sample_handling()

    train_x = []
    train_y = []
    test_x = []
    test_y = []

    for value in range(0, int(math.floor((1-test_size)*len(features)))):
        train_x.append(features[value][0])
        train_y.append(features[value][1])

    for value in range(0, int(math.ceil(test_size*len(features)))):
        test_x.append(features[-value][0])
        test_y.append(features[-value][1])

    return train_x, train_y, test_x, test_y
