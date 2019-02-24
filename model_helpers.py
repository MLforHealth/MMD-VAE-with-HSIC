import numpy as np
from matplotlib import pyplot as plt 
import math
from hyperparameters import *


# Convert a numpy array of shape [batch_size, height, width, 1] into a displayable array 
# of shape [height*sqrt(batch_size, width*sqrt(batch_size))] by tiling the images
def convert_to_display(samples):
    cnt, height, width = int(math.floor(math.sqrt(samples.shape[0]))), samples.shape[1], samples.shape[2]
    samples = np.transpose(samples, axes=[1, 0, 2, 3])
    samples = np.reshape(samples, [height, cnt, cnt, width])
    samples = np.transpose(samples, axes=[1, 0, 2, 3])
    samples = np.reshape(samples, [height*cnt, width*cnt])
    return samples


# helper function for getting batches
def next_batch(images, label1, label2, index_in_epoch, batch_size):
    start = index_in_epoch
    # increase the index in epoch by the batch size
    index_in_epoch += batch_size
    end = index_in_epoch
    if index_in_epoch > NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN:
      end = -1
      index_in_epoch = 0
     
    batch_label1 = label1[start:end]
    batch_label2 = label2[start:end]

    # randomly generate label to create null model
#     np.random.shuffle(batch_label)
    
    return images[start:end], batch_label1, batch_label2, index_in_epoch
  
def shuffle_data(images, label1, label2):
    perm = np.arange(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
    np.random.shuffle(perm)
    images = images[perm]
    label1 = label1[perm]
    label2 = label2[perm]
    return images, label1, label2