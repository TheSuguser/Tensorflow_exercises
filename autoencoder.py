import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

# Import training data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

# Training Parameters
learning_rate = 0.01
num_training = 1000
batch_size = 100

# Neural Network Parameters
num_hidden_1 = 256
num_hidden_2 = 128
num_input = 784 # Because figure size in dataset is 28*28

# tf Graph input
X = tf.placeholder(dtype = tf.float32, shape = ([None, num_input]))

# Weight
W = {
	'encoder_layer1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
	'encoder_layer2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
	'decoder_layer1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
	'decoder_layer2': tf.Variable(tf.random_normal([num_hidden_1, num_input]))
}

# Bias
b = {
	'encoder_layer1': tf.Variable(tf.random_normal([num_hidden_1])),
	'encoder_layer2': tf.Variable(tf.random_normal([num_hidden_2])),
	'decoder_layer1': tf.Variable(tf.random_normal([num_hidden_1])),
	'decoder_layer2': tf.Variable(tf.random_normal([num_input]))
}

# Define the encoder
def encoder(x):

	# In this demo I assume the activation is sigmoid. The first layer of encoder is shown below
	layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, W['encoder_layer1']), b['encoder_layer1']))
	# The second hidden layer of encoder
	layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, W['encoder_layer2']), b['encoder_layer2']))

	return layer2

# Define the decoder
def decoder(x):
	# The first layer of the decoder
	layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, W['decoder_layer1']), b['decoder_layer1']))
	# The second layer of the decoder
	layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, W['decoder_layer2']), b['decoder_layer2']))

	return layer2

encoder_output = encoder(X)
decoder_output = decoder(encoder_output)

y_star = decoder_output
y = X

# Define loss and optimizer
loss = tf.reduce_mean(tf.pow(y - y_star,2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)


init = tf.global_variables_initializer()
sess = tf.Session()

# Start training
# initialize the variables
sess.run(init)

for i in range(0, num_training):
	print("Training %d"%(i))

	batch_x, label = mnist.train.next_batch(batch_size)

	opti, l = sess.run([optimizer, loss],feed_dict ={X: batch_x})


# Testing
num_pic = 1
pic_input = np.empty((num_pic*28, num_pic*28))
pic_output = pic_input

batch_x, label = mnist.test.next_batch(num_pic)
digit_output = sess.run(decoder_output, feed_dict = {X: batch_x})

# Display the figures
pic_input = batch_x.reshape([28,28])
pic_output = digit_output.reshape([28,28])

# Show the figures
plt.figure(1,figsize = (num_pic,num_pic))
plt.imshow(pic_input, cmap = 'gray')

plt.figure(2,figsize = (num_pic,num_pic))
plt.imshow(pic_output, cmap = 'gray')

plt.show()










