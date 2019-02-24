import numpy as np
import tensorflow as tf
from hyperparameters import *

# Define some handy network layers
def lrelu(x, rate=0.1):
    return tf.maximum(tf.minimum(x * rate, 0), x)

def conv2d_lrelu(inputs, num_outputs, kernel_size, stride):
    conv = tf.contrib.layers.convolution2d(inputs, num_outputs, kernel_size, stride, 
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           activation_fn=tf.identity)
    conv = tf.contrib.layers.batch_norm(
          conv, center=True, scale=True,
          epsilon=1e-05, decay=0.9, updates_collections=None, fused=False)
    
    conv = lrelu(conv)
    
    return conv

def conv2d_t_relu(inputs, num_outputs, kernel_size, stride):
    conv = tf.contrib.layers.convolution2d_transpose(inputs, num_outputs, kernel_size, stride,
                                                     weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                     activation_fn=tf.identity)
    
    conv = tf.contrib.layers.batch_norm(
          conv, center=True, scale=True,
          epsilon=1e-05, decay=0.9, updates_collections=None, fused=False)
    
    conv = tf.nn.relu(conv)
    return conv

def fc_lrelu(inputs, num_outputs):
    fc = tf.contrib.layers.fully_connected(inputs, num_outputs,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           activation_fn=tf.identity)
    fc = lrelu(fc)
    return fc

def fc_relu(inputs, num_outputs):
    fc = tf.contrib.layers.fully_connected(inputs, num_outputs,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           activation_fn=tf.identity)
    fc = tf.nn.relu(fc)
    return fc

def max_pool2d(inputs, kernel_size):
    return tf.contrib.layers.max_pool2d(inputs, kernel_size)
  
def dropout(inputs):
    return tf.contrib.layers.dropout(inputs)
    