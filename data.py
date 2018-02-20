#!/usr/local/bin/python3
import csv
import numpy as np
import pickle
from collections import Counter

'''
Timestamp, Open, High, Low,	Close, Volume_(BTC), Timestamp, Open
'''
def sample_handling():

    featureset = []

    with open("data.csv", 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            featureset.append([row[1], row[2], row[3], row[4], row[5], row[6], row[7]])

        return featureset


def create_feature_sets_and_labels(test_size=0.1):
    features = sample_handling()

    features = np.array(features)

    testing_size = int(test_size * len(features))

    train_a = list(features[:,0][:-testing_size])
    train_b = list(features[:,1][:-testing_size])
    train_c = list(features[:,2][:-testing_size])
    train_d = list(features[:,3][:-testing_size])
    train_e = list(features[:,4][:-testing_size])
    train_f = list(features[:,5][:-testing_size])
    train_g = list(features[:,6][:-testing_size])

    test_a = list(features[:,0][-testing_size:])
    test_b = list(features[:,1][-testing_size:])
    test_c = list(features[:,2][-testing_size:])
    test_d = list(features[:,3][-testing_size:])
    test_e = list(features[:,4][-testing_size:])
    test_f = list(features[:,5][-testing_size:])
    test_g = list(features[:,6][-testing_size:])

    return [train_a, train_b, train_c, train_d, train_e, train_f, train_g, test_a, test_b, test_c, test_d, test_e, test_f, test_g]


if __name__ == '__main__':
    train = create_feature_sets_and_labels()
    #train_a , train_b , train_c , train_d , train_e , train_f , train_g , test_a , test_b , test_c , test_d , test_e , test_f , test_g = create_feature_sets_and_labels()
    for i in range(1,50):
        for l in range(0,7):
            print(train[l][i])
    #train_x, train_y, test_x, test_y = create_feature_sets_and_labels('positive.txt', 'negative.txt')
    # with open('sentiment_set.pickle', 'wb') as f:
    #     pickle.dump([train_x, train_y, test_x, test_y], f)
