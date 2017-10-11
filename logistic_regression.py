import tensorflow as tf 
import numpy as np 

# Import training data
from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Training parameter
learning_rate = 0.01
num_training = 20
batch_size = 100

# tf graph input
x = tf.placeholder(dtype = tf.float32, shape = ([None, 784]))
y = tf.placeholder(dtype = tf.float32, shape = ([None, 10]))

# Weight
W = tf.Variable(tf.zeros([784, 10]))

# Bias
b = tf.Variable(tf.zeros([None,10]))
# Model
y_pred = tf.nn.softmax(tf.add(tf.matmul(x,W),b)

# Cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pred), reduction_indice = 1))

# Using gradient descent to find the optimizer
optimizer = tf.training.GradientDescentOptimizer(learning_rate).minimize(cost)

# Start training
init = tf.global_variable_initializer()

sess = tf.Session()

sess.run(init)

for i in range(num_training):

	x,y = 