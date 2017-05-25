import pandas as pd
import tensorflow as tf
import os
import numpy as np

TRAIN_IMAGES_FILE = '../../.datasets/cifar-10/train_images.csv'
TRAIN_LABELS_FILE = '../../.datasets/cifar-10/train_labels.csv'
TEST_IMAGES_FILE = '../../.datasets/cifar-10/test_images.csv'
TEST_LABELS_FILE = '../../.datasets/cifar-10/test_labels.csv'

LOG_DIR = 'log/gr/'



"""
A basic 3d convolution layer, with args:
input - a 5-D tensor of shape 
    [#of_images, #image_width, #image_height, #image_depth, #of_features]
    example tensor shape: [-1, 32, 32, 3, 1]
filer_size - the size of the convolution filter
strides - the size of the strides to take
size_in - # of layers per image of input
size_out - # of layers per image of output
name - the name of the layer

the function returns the RELU activation layer of the convolution:
a 5-D tensor of shape [#of_images, #image_width, #image_height, #image_depth, #of_features]
where #of_layers is equal to the arg size_out

"""
def conv_layer(input, size_in, size_out, filter_size=5, strides=1, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([filter_size, filter_size, size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="b")
        conv = tf.nn.conv2d(input, w, strides=[1, strides, strides, 1], padding="SAME")
        act = tf.nn.relu(conv + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activation", act)
        return act
    
"""
A 3d max_pool layer with args:
input - a 5-D tensor of shape 
    [#of_images, #image_width, #image_height, #image_depth, #of_features]
    example tensor shape [-1, 32, 32, 3, 50]
kernel - a 5-D tensor of shape 
    [1, ksize1, ksize2, ksize3, 1]
    example tensor shape [1, 2, 2, 3, 1]
strides - a 5-D tensor of shape
    [1, strides1, strides2, 1, 1]
name - the name of the layer
    
this function outputs a 5-D tensor after the max_pool_3D operation  
"""
def max_pool(input, kernel, strides, name="max_pool"):
    with tf.name_scope(name):
        max_pool_3d = tf.nn.max_pool(input, ksize=kernel, strides=strides, padding="SAME")
        tf.summary.histogram("max_pooling", max_pool_3d)
        return max_pool_3d
    
"""
a fully connected layer with args:
input - a 2-D tensor of shape [#of_images, #of_features]
size_in - the number of features
size_out - the number of features of the output
name - the name of the layer

this function returns a 2-D tensor of shape [#of_images, #of_features_out]
where #of_features_out equals size_out
"""
def fc_layer(input, size_in, size_out, name="fully_connected_layer"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="b")
        act = tf.nn.relu(tf.matmul(input, w) + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act

"""
a dropout layer, that performs a simple dropout operation with args:
input - a 2-D tensor of shape [#of_images, #of_features]
keep_probability - a scalar the same type as the input, determins the probability of keeping each value
name - the name of the layer

outputs a 2-D tensor of the same shape as the input
"""
def dropout_layer(input, keep_probability, name="dropout"):
    with tf.name_scope(name):
        do = tf.nn.dropout(input, keep_probability)
        tf.summary.histogram("dropout", do)
        return do


"""
a function that returns a randomized batch of data/labels from the whole dataset/labelset.
arg:
dataset - a pd dataframe of all data
labelset - a pd dataframe of all labels
    // dataset on index i should correspond to the labelset index i
batch_size - the size of the needed random batch

returns a pair [data, labels] of data and labes of size batch_size
"""
def get_batch(dataset, labelset, batch_size=100):
    
    
    
    selector = np.concatenate((np.ones(batch_size),
                              np.zeros(dataset.shape[0] - batch_size)), axis = 0)

    np.random.shuffle(selector)
    selector = selector.astype('bool')

    
    batch_output = []
    
    batch_output.append(dataset.iloc[selector])
    batch_output.append(labelset.iloc[selector])
    
    return batch_output



def save_output(output):
	df = pd.DataFrame(output)
	df.to_csv(LOG_DIR + 'output.csv')




def cifar_10_model(dataset, labelset, learning_rate, momentum, numberOfSteps=5001):
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, shape=[None, 3072], name='x')
    y = tf.placeholder(tf.float32, shape=[None, 10], name='labels')
    
    #lets normalize our image vector
    x0 = tf.multiply(x, 1/255, name="normalizer")

    """
    We have to transform the 2-D tensor to a 5-D tensor for the 3D convolution, where:
     - 1-dim is the number of example images
     - 2-dim is the width of the image (32)
     - 3-dim is the height of the image(32)
     - 4-dim is the depth of the image (3, RGB)
     - 5-dim is the number of features, ATM = 1

    The input image is stored as RRRR...RGGGG...GBBBB...B: 
     - the first 1024 columns are 32x32 red values
     - the second 1024 columns are 32x32 green values
     - the third 1024 columns are 32x32 blue values
     
    Each line has to be transformed in the following way: 
        1 x 3072 > 1 x 3 x 1024 > 1 x 1024 x 3 > 1 x 32 x 32 x 3 > 1 x 32 x 32 x 3 x 1
    """
    
    x1 = tf.reshape(x0, [-1,3,1024])
    x2 = tf.transpose(x1, perm=[0,2,1])
    x3 = tf.reshape(x2, [-1,32,32,3])
    x4 = tf.image.rgb_to_grayscale(x3, name=None)
    
    tf.summary.image("input", x4, 10)

    x_image = x4
    
    """
    Time to define our model. 
    It will consist of 2 conv layers 2 max pool layers, 2 fc layers and 1 dropout layer
    sizes will vary in the following way:
    [-1, 32, 32, 3, 1]
    conv1
    [-1, 32, 32, 3, 50]
    max_pool
    [-1, 16, 16, 1, 50]
    conv2
    [-1, 16, 16, 1, 100]
    max_pool
    [-1, 8, 8, 1, 100]
    flatten
    [-1, 8*8*100 = 6400]
    fc_layer
    [-1, 1000]
    drouput
    [-1, 1000]
    fc_layer
    [-1, 10]
    """
    
    conv1 = conv_layer(x_image, 1, 32, filter_size=5, strides=1, name="conv_layer_1")
    
    mp1 = max_pool(conv1, [1,2,2,1], [1,2,2,1], name="max_pool_1")
    
    conv2 = conv_layer(mp1, 32, 64, filter_size=5, strides=1, name="conv_layer_2")
    
    mp2 = max_pool(conv2, [1,2,2,1], [1,2,2,1], name="max_pool_2")
    
    with tf.name_scope("flatten"):

    	flat = tf.reshape(mp2, [-1, 8 * 8 * 1 * 64]) # = 4096
   
    	tf.summary.histogram("flattened tensor", flat)

    fc1 = fc_layer(flat, 4096, 1000, name="fc_layer_1")
    
    #drop = dropout_layer(fc1, 0.5, name="dropout")
    
    logits = fc_layer(fc1, 1000, 10, name="fc_layer_2")

    
    # define cross entropy
    with tf.name_scope('xent'):
        xent = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y), name="x_ent")
        tf.summary.scalar('cross_entropy', xent)
    
    # define the trainig step using AdamOptimizer and a learning rate of 1E-4 
    with tf.name_scope('train'):
        train_step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(xent)

    # define accuracy
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)
    
    sess = tf.Session()
  
    merged_summary = tf.summary.merge_all()

    writer = tf.summary.FileWriter(LOG_DIR)
    
    sess.run(tf.global_variables_initializer())
    
    saver = tf.train.Saver()
    
    writer.add_graph(sess.graph)
    
    checkpointName = LOG_DIR + 'myCIFAR-10-grayscale_model'
    
    print('Trying to load previous model from: %s' %(LOG_DIR))
    try: 
        f = open(LOG_DIR + 'checkpoint', 'r')
        cp_path = f.readline()
        f.close()
        cp_path = cp_path[cp_path.find('"')+1 : cp_path.rfind('"')]
        cp_path = LOG_DIR + cp_path
        saver.restore(sess, cp_path)
        print('Model succesfully restored from: %s.' %(cp_path))

    except FileNotFoundError:
        print('Can not load model: no checkpoint found.')
        
        
    for i in range(numberOfSteps):
    
        batch = get_batch(dataset, labelset, batch_size=128)

        sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})

        if i % 5 == 0:
            [s, train_acc, output] = sess.run([merged_summary, accuracy, logits], feed_dict={x: batch[0], y: batch[1]})
            writer.add_summary(s, i)

        if i % 500 == 0:
            #train_acc = sess.run(accuracy, feed_dict={x: batch[0], y: batch[1]})
            print("Step %d, training accuracy %g" %(i, train_acc))
            save_output(output)

        if i % 1000 == 0 and i > 0:
            print('Saving checkpoint.')
            saver.save(sess, checkpointName, global_step=i)



if __name__ == '__main__':
	dataset = pd.DataFrame.from_csv(TRAIN_IMAGES_FILE, header=None, index_col=None)
	labelset = pd.DataFrame.from_csv(TRAIN_LABELS_FILE, header=None, index_col=None)

	learning_rate = 1E-4

	cifar_10_model(dataset, labelset, learning_rate, 0.9, numberOfSteps=5001)
