import tensorflow as tf
import numpy as np


##################################################################
# Higgins classifier-based disentaglement metrics implemantation
# For Mutual Information Gap (MIG) implementation, follow:
# https://github.com/rtqichen/beta-tcvae
##################################################################

# Load data 
GENERATED_LATENT_DIFF_DATASET = "hisc_wae_5000_diff_z.npz"
data = np.load(GENERATED_LATENT_DIFF_DATASET)

# Data schema
# {
#     "all_y": label for disentangled feature,
#     "diff_z": difference between generated latent representations 
# }

idx = np.random.permutation(len(data['all_y']))
all_x, all_y = data['diff_z'][idx], data['all_y'][idx]
# convert to one-hot
all_y = np.array([[1, 0] if i == 0 else [0, 1] for i in all_y])

train_size = int(len(data['all_y']) * .8)
train_x, train_y = all_x[:train_size], all_y[:train_size]
test_x , test_y = all_x[train_size:], all_y[train_size:]

learning_rate = 0.001
training_epochs = 2001
batch_size = 100
display_step = 200  

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 15 
n_classes = 2

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

def multilayer_perceptron(x, weights, biases):
    # Use tf.matmul instead of "*" because tf.matmul can change it's dimensions on the fly (broadcast)
    print( 'x:', x.get_shape(), 'W1:', weights['h1'].get_shape(), 'b1:', biases['b1'].get_shape())        
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1']) #(x*weights['h1']) + biases['b1']
    layer_1 = tf.nn.relu(layer_1)

    # Hidden layer with RELU activation
    print( 'layer_1:', layer_1.get_shape(), 'W2:', weights['h2'].get_shape(), 'b2:', biases['b2'].get_shape())        
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']) # (layer_1 * weights['h2']) + biases['b2'] 
    layer_2 = tf.nn.relu(layer_2)

    # Output layer with linear activation
    print( 'layer_2:', layer_2.get_shape(), 'W3:', weights['out'].get_shape(), 'b3:', biases['out'].get_shape())        
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out'] # (layer_2 * weights['out']) + biases['out']    
    print('out_layer:',out_layer.get_shape())

    return out_layer


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),    #15x256
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])), #256x256
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))  #256x1
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),             #256x1
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),             #256x1
    'out': tf.Variable(tf.random_normal([n_classes]))              #10x1
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.initialize_all_variables()

# helper function for getting batches
def next_batch(train_data, label1, index_in_epoch, batch_size):
    start = index_in_epoch
    # increase the index in epoch by the batch size
    index_in_epoch += batch_size
    end = index_in_epoch
    if index_in_epoch > train_size:
      end = -1
      index_in_epoch = 0
     
    batch_label1 = label1[start:end]

    # randomly generate label to create null model
#     np.random.shuffle(batch_label)
    
    return train_data[start:end], batch_label1, index_in_epoch
  
def shuffle_data(images, label1):
    idx = np.random.permutation(train_size)
    images = images[idx]
    label1 = label1[idx]
    
    return images,label1

# Launch the graph
index_in_epoch = 0
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        batch_x, batch_y, index_in_epoch = next_batch(train_x, train_y, index_in_epoch, batch_size)
        if index_in_epoch == 0:
            train_x, train_y = shuffle_data(train_x, train_y)
            batch_x, batch_y, index_in_epoch = next_batch(train_x, train_y, index_in_epoch, batch_size)

        _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                    y: batch_y})

        # Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(c))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # To keep sizes compatible with model
    print ("Accuracy:", accuracy.eval({x: test_x, y: test_y}))