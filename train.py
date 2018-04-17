#!/usr/local/bin/python3
import tensorflow as tf
import numpy as np
from data import create_feature_sets_and_labels
import numpy as np
import matplotlib.pyplot as plt

train_x, train_y, test_x, test_y = create_feature_sets_and_labels()

n_nodes_hl1 = 6
n_nodes_hl2 = 6
n_nodes_hl3 = 6

n_classes = 1
batch_size = 100

x = tf.placeholder('float', [None, len(train_x[0])])
y = tf.placeholder('float', [None, 1])
z = tf.placeholder('float', [None, 1])

def neural_network_model(data):

    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases' :tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases' :tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer =   {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases' :tf.Variable(tf.random_normal([n_classes]))}


    #do the linear combinations now
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

    #print(Tensor.eval(output))
    return output


def train_neural_network():
    #print(train_x)
    prediction = neural_network_model(train_x)
    cost = tf.reduce_sum(tf.square(tf.subtract(prediction, train_y)))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    #cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    hm_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #print(prediction.eval())

        for epoch in range(hm_epochs):
            epoch_loss = 0

            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size

                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                #_, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                _, c = sess.run([optimizer, cost])
                #prediction = neural_network_model(x)
                #sess.run(z, feed_dict={z: y})
                epoch_loss += c
                i += batch_size

            print('Epoch', epoch+1, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        print("train_y = ", train_y[:])
        print("prediction = ", prediction.eval()[:])

        value = (tf.divide(tf.subtract(prediction, train_y), prediction))
        acc = value.eval()
        accuracies = tf.reduce_mean(acc)
        print(accuracies.eval()) #We want this to be as close to 0 as possible!

        nums = []

        for x in range(len(acc)):
            nums.append(x)

        '''
        plotting
        '''

        '''
        plt.plot(prediction.eval()[:], acc)
        plt.xlabel('output')
        plt.ylabel('percent difference')
        plt.title('percent difference vs output')
        plt.savefig('percentdiffvsoutput.png')
        '''


        fig, ax1 = plt.subplots()
        ax1.plot(nums, acc[:], 'b-')
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Percent Difference', color='b')
        ax1.tick_params('y', colors='b')
        ax2 = ax1.twinx()
        ax2.plot(nums, train_y[:], 'r-')
        ax2.set_ylabel('Expected Output', color='r')
        ax2.tick_params('y', colors='r')

        fig.tight_layout()

        plt.title("Percent Difference Compared With Training Data")
        plt.show()


        # plt.plot(train_y[:], prediction.eval()[:], 'b', numsx, numsy, 'r')
        # plt.xlabel("Expected Output")
        # plt.ylabel("Network Output")
        # plt.title("Network Output vs Expected Output")
        # plt.legend(["Network output vs Expected output", "Y = X"],4)
        #
        # plt.show()






        #correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        #accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        #print('Accuracy:', accuracy.eval({x:test_x, y:test_y}))

train_neural_network()
