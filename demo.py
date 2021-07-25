import tensorflow.compat.v1 as tf
import os
from DataLoader import Data_load
#from test_lstm_classification import dynamicRNN

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


tf.compat.v1.disable_eager_execution()

Seq_Len = 350
n_hidden = 64
n_classes = 5 

x = tf.placeholder("float", [None, Seq_Len, 68])
y = tf.placeholder("float", [None, n_classes])
seqlen = tf.placeholder(tf.int32, [None])
learning_rate = 0.002788
weights = {'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))}
biases  = {'out': tf.Variable(tf.random_normal([n_classes]))}

path = './model/'
trainset = Data_load()
testset = Data_load(File = "./data/test/00")

Pre_Train = dynamicRNN(x, seqlen, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Pre_Train, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
Correct_Pre = tf.equal(tf.argmax(Pre_Train, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(Correct_Pre, tf.float32))
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    if os.path.exists(path + 'checkpoint'):         #判断模型是否存在
        saver.restore(sess, path + 'my-model')    #存在就从模型中恢复变量
        print("loaded")
    
    batch_x, batch_y, batch_seqlen = trainset.returnData()   
    acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
    loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
    print("Minibatch Loss= " + "{:.5f}".format(loss) + \
          ", Training Accuracy= " + "{:.4f}".format(acc))        
    Tacc = sess.run(accuracy, feed_dict={x: testset.data, y: testset.labels, seqlen: testset.seqlen})
    print("Testing Accuracy:", Tacc)
    mark = [0, 27, 54, 81, 107, 117]
    testacc = []
    for i in range(5):
        st = mark[i]
        ed = mark[i + 1] - 1
        Testacc = sess.run(accuracy, feed_dict={x: testset.data[st: ed], y: testset.labels[st: ed], seqlen: testset.seqlen[st: ed]})
        testacc.append(Testacc)
        print("Class:00{}  Test Accuracy:".format(i), Testacc)

