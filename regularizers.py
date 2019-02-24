import numpy as np
import tensorflow as tf
from hyperparameters import *

# MMD implementation
def MMD(X, Y, kernel='Gaussian', bandwidth=0):
    
    if bandwidth <= 0: # median heuristic
        bandwidth = comp_med(tf.concat([X,Y],0))

    XX = compute_gram(X,X,bandwidth,kernel)
    YY = compute_gram(Y,Y,bandwidth,kernel)
    XY = compute_gram(X,Y,bandwidth,kernel)
    XX = tf.reduce_mean(XX)
    YY = tf.reduce_mean(YY)
    XY = tf.reduce_mean(XY)
    
    return XX + YY - 2*XY # biased V-stats

# HSIC implementation

def HSIC(X, Y, kernel='Gaussian', bandwidthX=0, bandwidthY=0, normalized=True): # set normalized to False
    if tf.keras.backend.ndim(X) == 1:
        X = tf.expand_dims(X,axis=1)
    if tf.keras.backend.ndim(Y) == 1:
        Y = tf.expand_dims(Y,axis=1)
    n = tf.shape(X)[0]
    
    if bandwidthX == 0:
        bandwidthX = comp_med(X,kernel)
    if bandwidthY == 0:
        bandwidthY = comp_med(Y,kernel)
        
    K = compute_gram(X,X,bandwidthX,kernel)
    L = compute_gram(Y,Y,bandwidthY,kernel)
    H = tf.eye(n) - (tf.ones([n,n]) / tf.cast(n,tf.float32))
    Kc = tf.matmul(tf.matmul(H,K),H)
    trace = tf.reduce_sum(L*(tf.transpose(Kc)))
    if normalized:
        HKH = tf.norm(tf.matmul(tf.matmul(H,K),H),ord='fro',axis=[0,1])
        HLH = tf.norm(tf.matmul(tf.matmul(H,L),H),ord='fro',axis=[0,1])
        return trace / (HKH*HLH)
    else:
        return trace / tf.cast((n*n),tf.float32)


# 
# Helper functions for computing MMD and HSIC 
#
def reparameterize(mean, logvar, random=True):
    if random:
        eps = tf.random_normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * .5) + mean
    else: # deterministic
        return mean
    
def push_forward(encoder, train_x):
    mean, logvar = tf.split(encoder(train_x,z_dim), num_or_size_splits=2, axis=1)
    return reparameterize(mean,logvar)

def comp_med(X,kernel='Gaussian'):
    H = compute_diff(X,X)
    H = tf.where(tf.greater(H,0))
    if tf.shape(H)[0] == 0:
        return 1
    else:
        h = tf.contrib.distributions.percentile(H, 50.0)
        if kernel == 'Gaussian':
            return tf.sqrt(tf.cast(h/2,tf.float32)) 
            #return np.sqrt(0.5 * h) / np.log(X.shape[0]+1)
        else:
            return tf.sqrt(tf.cast(h,tf.float32))

def compute_diff(X,Y):
    XY = X[:,tf.newaxis,:] - Y
    out = tf.einsum('ijk,ijk->ij',XY,XY)
    return out
    
def compute_gram(X,Y, bandwidth, kernel='Gaussian'):
    H = compute_diff(X,Y)
    if kernel == 'Gaussian':
        return tf.exp(- H / 2 / tf.cast(bandwidth**2, tf.float32)) # (x_size, y_size)
    else: # inverse multiquadratics kernel
        # return c / (c + H) 
        return tf.pow(tf.cast(bandwidth**2,tf.float32) + H, -0.5)

def gather_cols(params, indices, name=None):
    """Gather columns of a 2D tensor.

    Args:
        params: A 2D tensor.
        indices: A 1D tensor. Must be one of the following types: ``int32``, ``int64``.
        name: A name for the operation (optional).

    Returns:
        A 2D Tensor. Has the same type as ``params``.
    """
    with tf.op_scope([params, indices], name, "gather_cols") as scope:
        # Check input
        params = tf.convert_to_tensor(params, name="params")
        indices = tf.convert_to_tensor(indices, name="indices")
        try:
            params.get_shape().assert_has_rank(2)
        except ValueError:
            raise ValueError('\'params\' must be 2D.')
        try:
            indices.get_shape().assert_has_rank(1)
        except ValueError:
            raise ValueError('\'params\' must be 1D.')

        # Define op
        p_shape = tf.shape(params)
        p_flat = tf.reshape(params, [-1])
        i_flat = tf.reshape(tf.reshape(tf.range(0, p_shape[0]) * p_shape[1],
                                       [-1, 1]) + indices, [-1])
        return tf.reshape(tf.gather(p_flat, i_flat),
                          [p_shape[0], -1])
