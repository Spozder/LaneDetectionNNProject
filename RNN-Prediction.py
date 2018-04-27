import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from sklearn.model_selection import train_test_split
from skimage import transform

original_images_raw = pickle.load(open("bridge.p", "rb"))[0:52850]
O = np.array([transform.resize(i, (40, 80)) for i in original_images_raw])

predict_images_raw = pickle.load(open("bridge_CNN_labels.p", "rb" ))[0:75]
print(len(predict_images_raw))
I = np.array([transform.resize(i, (40, 80)) for i in predict_images_raw]).reshape(-1, 3, 3200)


memory_size = 3
batch_size = 17

#Size of our input image. e.g. number of features
number_of_features = 40 * 80

steps_num = 1000

tf.reset_default_graph()

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
saver = tf.train.Saver()

# Cell used to predicting
with tf.Session() as sess:
    saver.restore(sess, "./tmp/rnn_model_RNN_SIGMOID/rnn_model")
    # input data used to predict, format:
    x_input = I
    print(x_input[0][0][3000:3020])
    # The output should be a 1-D array whose size is height * width. 
    # In order to display it, it needs reshpe
    y_pred = sess.run(outputs, feed_dict={X:x_input})
    print(y_pred[0][0][3000:3020])
    
    n = 55

    plt.figure()
    plt.imshow(O[n])
    
    I = I.reshape(-1, 40, 80)

    plt.figure()
    plt.imshow(I[n])
    
    P = y_pred.reshape(-1, 40, 80)
    pickle.dump(P,open('bridge_RNN_Sigmoid_labels.p', "wb" ))

    plt.figure()
    plt.imshow(P[n])