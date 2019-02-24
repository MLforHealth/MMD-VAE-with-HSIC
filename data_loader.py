import numpy as np 
from hyperparameters import *

##################################################################
# Loading data
# Using dsprite data as example, assume dataset name is dsprites.npz
##################################################################

np.random.seed(6)

raw_data = np.load(DATASET, encoding='bytes')
imgs = raw_data['imgs']
latents_values = raw_data['latents_values']
metadata = raw_data['metadata'][()]

# Define number of values per latents and functions to convert to indices
latents_sizes = metadata[b'latents_sizes']
latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                np.array([1,])))

def latent_to_index(latents):
  return np.dot(latents, latents_bases).astype(int)

def sample_latent(size=1):
  samples = np.zeros((size, latents_sizes.size))
  for lat_i, lat_size in enumerate(latents_sizes):
    samples[:, lat_i] = np.random.randint(lat_size, size=size)

  return samples

def load_data():
    # Sample latents randomly 
    latents_sampled = sample_latent(size=SAMPLE_SIZE)
    # Select images
    indices_sampled = latent_to_index(latents_sampled)
    imgs_sampled = imgs[indices_sampled]
    latents_values_sampled = latents_values[indices_sampled]

    data_total_size = imgs_sampled.shape[0]
    train_size = int(data_total_size * .8)
    test_size = data_total_size - train_size

    print("Train_size: {} \nTest_size: {}".format(train_size, test_size))

    d_train_x = imgs_sampled[:train_size]/255.
    # position_y
    d_train_y1 = latents_values_sampled[:train_size][:,5]
    # position_x 
    d_train_y2 = latents_values_sampled[:train_size][:,4]
    d_test_x = imgs_sampled[train_size:]/255.
    # position_x
    d_test_y1 = latents_values_sampled[train_size:][:,5]
    # position_x
    d_test_y2 = latents_values_sampled[:train_size][:,4]

    return d_train_x, d_train_y1, d_train_y2, d_test_x, d_test_y1, d_test_y2