#!/usr/local/bin/python3
import csv
import numpy as np
import pickle
from collections import Counter
import math

'''
Timestamp, Open, High, Low,	Close, Volume_(BTC), Volume_(Currency)
'''
def sample_handling():

    featureset = []

    with open("data.csv", 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        current_high = 0.0
        current_low = 0.0
        current_open = 0.0
        current_close = 0.0
        current_volume_btc = 0.0
        current_volume_currency = 0.0
        current_net_change = 0.0 #will become difference between current_open and current_close

        prev_high = 0.0
        prev_low = 0.0
        prev_open = 0.0
        prev_close = 0.0
        prev_volume_btc = 0.0
        prev_volume_currency = 0.0
        prev_delta_high = 0.0

        delta_high = 0.0
        delta_low = 0.0
        delta_open = 0.0
        delta_close = 0.0
        delta_volume_btc = 0.0
        delta_volume_currency = 0.0

        counter = 0

        indicator = 0 # -1 for bad, 0 for ok, 1 for good

        for row in reader:
            #set current variables to current values from row
            current_high = float(row[2])
            current_low = float(row[3])
            current_open = float(row[1])
            current_close = float(row[4])
            current_volume_btc = float(row[5])
            current_volume_currency = float(row[6])
            if counter > 0:
                #calculated delta values
                delta_high = (current_high - prev_high)/prev_high
                delta_low = (current_low - prev_low)/prev_low
                delta_open = (current_open - prev_open)/prev_open
                detla_close = (current_close - prev_close)/prev_open
                delta_volume_btc = (current_volume_btc - prev_volume_btc)/prev_volume_btc
                delta_volume_currency = (current_volume_currency - prev_volume_currency)/prev_volume_currency
                #calculate current change values
                current_net_change = (current_close - current_open)/current_open
                #append the vector the the list featureset
                #featureset.append([delta_high, delta_low, delta_open, delta_close, delta_volume_btc, delta_volume_currency, current_net_change])
                #assign indicator here
                if prev_delta_high <= 0:
                    indicator = -1
                elif prev_delta_high > 0 and prev_delta_high < 1:
                    indicator = 0
                else:
                    indicator = 1
                featureset.append([[prev_delta_high, prev_high], [indicator, current_high]])

            #set previous values to current so that they are ready for the next loop through
            prev_high = current_high
            prev_low = current_low
            prev_open = current_open
            prev_close = current_close
            prev_volume_btc = current_volume_btc
            prev_volume_currency = current_volume_currency
            prev_delta_high = delta_high

            #increment counter
            counter += 1

        return featureset


def create_feature_sets_and_labels(test_size=0.1):
    features = sample_handling() #2d array

    # features = np.array(features)
    #
    # testing_size = int(test_size * len(features))
    #
    # train_x = list(features[:,0][:-testing_size])
    # train_y = list(features[:,1][:-testing_size])
    #
    # test_x = list(features[:,0][-testing_size:])
    # test_y = list(features[:,1][-testing_size:])

    train_x = []
    train_y = []
    test_x = []
    test_y = []

    for value in range(0, int(math.floor((1-test_size)*len(features)))):
        train_x.append(features[value][0])
        train_y.append(features[value][1])

    for value in range(0, int(math.ceil(test_size*len(features)))):
        test_x.append(features[value][0])
        test_y.append(features[value][1])

    print(test_x)

    return train_x, train_y, test_x, test_y
