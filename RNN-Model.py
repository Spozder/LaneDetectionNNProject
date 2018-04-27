import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from skimage import transform


def prepare_data(train_data, train_label, memory_unit, batch_size):
    number, height, width, channel = \
    train_data.shape[0], train_data.shape[1], train_data.shape[2], train_data.shape[3]

    number_of_mem_unit = number // memory_unit

    # let the data become from (1200, 80, 160, 1) to (40, 30, 80 * 160)
    sample_number = number_of_mem_unit * memory_unit
    unit_train_data = train_data[0:sample_number,:,:,:].reshape(-1, batch_size, memory_unit, height * width)
    
    # label shift to right by one step
    unit_train_label = train_label[0:sample_number,:,:,:].reshape(-1, batch_size, memory_unit, height * width)
    
    return iter(unit_train_data), iter(unit_train_label)


datalist = pickle.load(open("CNN_road_labels.p", "rb" ))[0:52850]
data = np.array([transform.resize(i, (40, 80)) for i in np.clip(np.array(datalist), 0, 1)])
labellist = pickle.load(open("road_labels.p", "rb" ))[0:52850]
label = np.array([transform.resize(i, (40, 80)) for i in labellist])

# label = label / 255

print(np.max(data))
print(np.max(label))

train_ratio = 0.8
memory_size = 3
batch_size = 17
train_data, test_data, train_label, test_label = \
train_test_split(data, label, train_size = train_ratio,
                              test_size = 1 - train_ratio,
                              shuffle = False)

input_data_iter, input_label_iter = prepare_data(train_data, train_label, 
                                                memory_size, batch_size)


# Do training
learning_rate = 0.001

#Size of our input image. e.g. number of features
number_of_features = 40 * 80

steps_num = 10000

#input data
X = tf.placeholder(tf.float32, [None, memory_size, number_of_features])
#data label
Y = tf.placeholder(tf.float32, [None, memory_size, number_of_features])

#RNN cells
cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicRNNCell(number_of_features, 
                                                                activation=tf.sigmoid), 
                                                                input_keep_prob=0.7, 
                                                                output_keep_prob=1, 
                                                                state_keep_prob=1)

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

#loss
loss = tf.losses.mean_squared_error(outputs, Y)

optimizer = tf.train.AdamOptimizer()

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for iteration in range(steps_num):
        try:
            train_data, train_labels = next(input_data_iter), next(input_label_iter)
        except StopIteration:
            print('Stopping iteration - no more input data')
            break
        sess.run(train, feed_dict = {X:train_data, Y:train_labels})
        error = loss.eval( feed_dict = {X:train_data, Y:train_labels})
        print("iteration {}, error is {}".format(iteration, error))
        print("\n")
    saver.save(sess, save_path="./tmp/rnn_model_RNN_SIGMOID/rnn_model")