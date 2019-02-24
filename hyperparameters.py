import numpy as np

batch_size = 512
z_dim = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 4000 
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 1000
SAMPLE_SIZE = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN + NUM_EXAMPLES_PER_EPOCH_FOR_TEST
Lambda1 = 1 # mmd
Lambda2 = 0.002 # HSIC, dep axis
Lambda3 = 0.05 # HSIC, rest of axis
prob_to_hsic = 100 # probability of applying HISC in regularization
steps = 20001
K1 =  'IMQ' # MMD kernel
K2 =  'Gaussian' # HSIC kernel
Sigma = 1
bandwidth1 = np.sqrt(z_dim) * Sigma # or -1 # IMQ for larger mmd
bandwidth2 = 0
DATASET = "dsprites.npz"
LEARN_RATE = 1e-3
