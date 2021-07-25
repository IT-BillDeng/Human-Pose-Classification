#LSTM_CLASSIFICATION
#UTF-8

from __future__ import print_function

import tensorflow.compat.v1 as tf
# import tensorflow as tf
import random
import math
import os
import numpy as np

from DataLoader import Data_load

tf.compat.v1.disable_eager_execution()

path = './model'

learning_rate = 0.00268
training_iters = 100000
batch_size = 2

Seq_Len = 350
n_hidden = 64
n_classes = 5 

x = tf.placeholder("float", [None, Seq_Len, 68])
y = tf.placeholder("float", [None, n_classes])
seqlen = tf.placeholder(tf.int32, [None])

weights = {'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))}
biases  = {'out': tf.Variable(tf.random_normal([n_classes]))}

trainset = Data_load()
testset = Data_load(File = "./data/test/00")

def dynamicRNN(x, seqlen, weights, biases):
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)

    outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seqlen)

    batch_size = tf.shape(outputs)[0]
    print(batch_size)
    index = tf.range(0, batch_size) * Seq_Len + (seqlen - 1)
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)
    Ans = tf.matmul(outputs, weights['out']) + biases['out']
    print(Ans.shape)
    return Ans

Pre_Train = dynamicRNN(x, seqlen, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Pre_Train, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
Correct_Pre = tf.equal(tf.argmax(Pre_Train, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(Correct_Pre, tf.float32))


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 1
    Tacc_Max = 0
    while step * batch_size < training_iters:
        batch_x, batch_y, batch_seqlen = trainset.returnData()
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y,seqlen:batch_seqlen})
        if not (step % 100):          
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})           
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.5f}".format(loss) + \
                  ", Training Accuracy= " + "{:.4f}".format(acc))
            test_data = testset.data
            test_label = testset.labels
            test_seqlen = testset.seqlen
            
            Tacc = sess.run(accuracy, feed_dict={x: test_data, y: test_label, seqlen: test_seqlen})
            
            # if Tacc>=0.65:
            #     saver = tf.train.Saver()
            #     saver.save(sess, path+"/my-model")
            #     break
            if Tacc >= Tacc_Max:
                saver = tf.train.Saver()
                saver.save(sess, path+"/my-model")
                Tacc_Max = Tacc
            print("Testing Accuracy:", Tacc, "Max Accuracy", Tacc_Max)

        step += 1
    print("Optimization Finished!")