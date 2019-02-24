import tensorflow as tf
import numpy as np
from layers import *
from regularizers import *
from hyperparameters import *
from model_helpers import *
from data_loader import load_data
import time

config = tf.ConfigProto()
# Turn off this option if no gpu and remove device in encoder and decoder
config.gpu_options.allow_growth = True
tf.reset_default_graph()

d_train_x, d_train_y1, d_train_y2, d_test_x, d_test_y1, d_test_y2 = load_data()

##################################################################
# Define model 
##################################################################

def encoder(x, z_dim, reuse=False):
  with tf.device('/gpu:0'):
    with tf.variable_scope('encoder') as en:
        if reuse:
            en.reuse_variables()
        
        conv1 = conv2d_lrelu(x, 32, 4, 2)
        pool1 = max_pool2d(conv1, [2, 2])
#         drop1 = dropout(pool1)
        
        conv2 = conv2d_lrelu(pool1, 64, 4, 2)
        pool2 = max_pool2d(conv2, [2, 2])
#         drop2 = dropout(pool2)

        conv3 = conv2d_lrelu(pool2, 128, 4, 1)
        pool3 = max_pool2d(conv3, [2,2])
        
        conv4 = conv2d_lrelu(pool3, 256, 4, 1)
        
        print(conv4)
        flat_z = tf.reshape(conv4, [-1, np.prod(conv4.get_shape().as_list()[1:])])
       
        fc1 = tf.contrib.layers.fully_connected(flat_z, 256, activation_fn=tf.nn.relu)
        
        return tf.contrib.layers.fully_connected(fc1, z_dim * 2, activation_fn=tf.identity)
  
def decoder(z, reuse=False):
  with tf.device('/gpu:0'):
    with tf.variable_scope('decoder') as vs:
        if reuse:
            vs.reuse_variables()
        
        fc2 = tf.contrib.layers.fully_connected(z, 512, activation_fn=lrelu)

        fc3 = fc_relu(fc2, 2*2*256)
        fc3 = tf.reshape(fc3, tf.stack([tf.shape(fc3)[0], 2, 2, 256]))
      
        deconv1 = conv2d_t_relu(fc3, 256, 4, 2)
        deconv2 = conv2d_t_relu(deconv1,128, 4, 2)
        deconv3 = conv2d_t_relu(deconv2, 64, 4, 2)
        deconv4 = conv2d_t_relu(deconv3, 32, 4, 2)
        deconv5 = tf.contrib.layers.convolution2d_transpose(deconv4, 1, 4, 2,
                                                     weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                     activation_fn=tf.identity)
        return tf.nn.sigmoid(deconv5)
              

##################################################################
# Build the computation graph for training
##################################################################
 
train_x = tf.placeholder(tf.float32, shape=[None, 64, 64, 1])
train_y1 = tf.placeholder(tf.float32, shape=[None])
train_y2 = tf.placeholder(tf.float32, shape=[None])

train_z = push_forward(encoder, train_x)
train_xr = decoder(train_z)

# Build the computation graph for generating samples# Build 
gen_z = tf.placeholder(tf.float32, shape=[None, z_dim])
gen_x = decoder(gen_z, reuse=True)

pretrained_mean, pretrained_var = tf.split(encoder(train_x, z_dim, reuse=True), num_or_size_splits=2, axis=1)

# Compare the generated z with true samples from a standard Gaussian, and compute their MMD distance
true_samples = tf.random_normal([batch_size, z_dim],stddev=Sigma)
loss_mmd = MMD(true_samples, train_z, kernel=K1, bandwidth=bandwidth1)

loss_nll = tf.reduce_mean(tf.square(train_xr - train_x))
hsic_signal = tf.placeholder(tf.bool)  # placeholder for a HISC toggle
hsic_trigger = tf.cond(tf.equal(hsic_signal, tf.constant(True)), lambda: tf.constant(1, tf.float32), lambda: tf.constant(0, tf.float32))

# apply lambda2 on the dependent feature axis and discriminate all other axis with lambda3
first_axis_hsic =  HSIC(gather_cols(train_z, [0]), train_y1, normalized=True)
second_axis_hsic = HSIC(gather_cols(train_z, [1]), train_y2, normalized=True)

# for HSIC with 1 side information
# other_axis_hsic =  HSIC(gather_cols(train_z, list(range(1,z_dim))), train_y1, normalized=True) 
other_axis_hsic =  HSIC(gather_cols(train_z, list(range(2,z_dim))), train_y1, normalized=True) + HSIC(gather_cols(train_z, list(range(2,z_dim))), train_y2, normalized=True)

loss_hsic = Lambda2 * (first_axis_hsic + second_axis_hsic) - Lambda3 * other_axis_hsic 
loss_hsic = hsic_trigger * loss_hsic

loss = loss_nll + Lambda1 * loss_mmd - loss_hsic
trainer = tf.train.AdamOptimizer(LEARN_RATE).minimize(loss)


##################################################################
# Start training session
##################################################################
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
mmd_list = []
o_loss_list = []
rec_loss_list = []
first_hisc_list = []
other_hisc_list = []
steps_list = []

# TODO implement tf saving model
# saver = tf.train.Saver() 

tf.set_random_seed(1109)

# using median heuristic bandwidth and HSIC
start_time = time.time()
index_in_epoch = 0
for i in range(steps):
    
    batch_x, batch_y1, batch_y2, index_in_epoch = next_batch(d_train_x, d_train_y1, d_train_y2, index_in_epoch, batch_size)
    if index_in_epoch == 0:
      d_train_x, d_train_y1, d_train_y2 = shuffle_data(d_train_x, d_train_y1, d_train_y2)
      batch_x, batch_y1, batch_y2, index_in_epoch = next_batch(d_train_x, d_train_y1, d_train_y2, index_in_epoch, batch_size)
    
    batch_x = batch_x.reshape(-1, 64, 64, 1)
    use_hsic = False
    if np.random.random() <= prob_to_hsic:
        use_hsic = True
    
    _, o_loss, nll, mmd, f_hsic, s_hsic, o_hsic, reconstr = sess.run([trainer, loss, loss_nll, loss_mmd, first_axis_hsic, second_axis_hsic, other_axis_hsic, train_xr], feed_dict={
                  train_x: batch_x, train_y1: batch_y1, train_y2: batch_y2, hsic_signal: use_hsic})

    if i % 1000 == 0:
        print("Using hsic:", use_hsic)
        print("epoch: {}, Overall loss is {}, recon loss is {}, mmd loss is {}, f_hsic is {}, s_hsic is {}, other hsic is {}".format(
            i, o_loss, nll, mmd, f_hsic, s_hsic, o_hsic))

        elapsed_time = time.time() - start_time
        print("time elapsed: {0:.2f}s".format(elapsed_time))
        start_time = time.time()
        # storing data for plot 
        mmd_list += [mmd]
        o_loss_list += [o_loss]
        rec_loss_list += [nll]
        steps_list += [i]
#         first_hisc_list += [f_hsic]
#         other_hisc_list += [o_hsic]
        
    if i % 5000 == 0:
        # feed in test image to get generated mmd loss
        test_x, test_y1, test_y2 = d_test_x[:batch_size], d_test_y1[:batch_size], d_test_y2[:batch_size]
        test_x = test_x.reshape(-1, 64, 64, 1)
        samples, gen_mmd, my_z= sess.run([gen_x, loss_mmd, pretrained_mean],  feed_dict={gen_z: np.random.normal(size=(49, z_dim)), train_x: test_x})
        plt.imshow(convert_to_display(samples), cmap='Greys_r', interpolation='nearest')
        plt.grid()
        plt.savefig('{}_steps.png'.format(i))
        print("generated mmd loss: {}, my_z: {}".format(gen_mmd, my_z[0]))


##################################################################
# Generated data for Higgins disentanglement metrics
##################################################################

np.random.seed(200)
# generate sample images for height at first axis
all_y = []
all_x1 = []
all_x2 = []
y = 0
for i in range(500):
  dep_axis = np.random.normal()

  rd1 = np.random.normal(size=(1, z_dim-1))[0]
  rd2 = np.random.normal(size=(1, z_dim-1))[0]
  v1 = np.concatenate((np.array([dep_axis]), rd1))
  v2 = np.concatenate((np.array([dep_axis]), rd2))
  v_samples = sess.run(gen_x,  feed_dict={gen_z: [v1, v2]})
  sim1 = v_samples[0].reshape((64, 64))
  sim2 = v_samples[1].reshape((64, 64))
  all_x1.append(sim1)
  all_y.append(y)
  all_x2.append(sim2)

print('Finihsed generating for the first axis.')

# generate sample images for position_x at second axis
y = 1
for i in range(500):
#   if i % 100 == 0:
#     print(i)
  dep_axis = np.random.normal()

  rd1 = np.random.normal(size=(1, z_dim-1))[0]
  rd2 = np.random.normal(size=(1, z_dim-1))[0]
  v1 = np.concatenate((np.array([rd1[0]]), np.array([dep_axis]), rd1[1:]))
  v2 = np.concatenate((np.array([rd1[0]]), np.array([dep_axis]), rd2[1:]))
  v_samples = sess.run(gen_x,  feed_dict={gen_z: [v1, v2]})
  sim1 = v_samples[0].reshape((64, 64))
  sim2 = v_samples[1].reshape((64, 64))
  all_x1.append(sim1)
  all_x2.append(sim2)
  all_y.append(y)

print('Finihsed generating for the second axis.')

all_x1 = np.array(all_x1)
all_x2 = np.array(all_x2)
all_y = np.array(all_y)

print(all_x1.shape, all_x2.shape, all_y.shape)
np.savez("hisc_wae_{}_sample_metrics".format(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN), all_x1=all_x1, all_x2=all_x2, all_y=all_y)

x1_z = sess.run(pretrained_mean,  feed_dict={train_x: all_x1.reshape(-1, 64, 64, 1)})
x2_z = sess.run(pretrained_mean,  feed_dict={train_x: all_x2.reshape(-1, 64, 64, 1)})
diff = x2_z - x1_z

np.savez("hisc_wae_{}_diff_z".format(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN), diff_z=diff, all_y=all_y)
