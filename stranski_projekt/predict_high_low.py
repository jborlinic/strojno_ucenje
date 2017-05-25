import tensorflow as tf
import numpy as np


"""
    Parameters
"""

LOAD_MODEL = True
NUM_OF_STEPS = 1000001
DATA_DIR = '../.datasets/high_low/'
LEARNING_RATE = 1E-4
TASK = 'Predict' #Predict or Train
MODEL_NUMBER = 3

"""
  1: 
    -reshape 
    -conv layer 1-32
    -flatten -224
    -dropout 0.5
    -logits 224-2

  2:
    -normalization
    -reshape 
    -conv layer 1-32
    -conv layer 32-128
    -flatten -896
    -fc layer 896-224
    -dropout 0.5
    -logits 224-2

  3:
    -normalization
    -fc layer 7-512
    -dropout 0.5
    -fc layer 512-224
    -logits 224-2

  4:
    -reshape 
    -conv layer 1-32
    -flatten -224
    -dropout 0.3
    -logits 224-2

  5:
    -logits 7-2
"""


LOGDIR = 'log/%d/' %MODEL_NUMBER


def csv_to_numpy_array(filePath, delimiter=";"):
    return np.genfromtxt(filePath, delimiter=delimiter, dtype=np.float)


def import_data():
    print("loading training data")
    trainX = csv_to_numpy_array("%strainX.csv" %DATA_DIR)
    trainY = csv_to_numpy_array("%strainY.csv" %DATA_DIR)
    print("loading test data")
    testX = csv_to_numpy_array("%stestX.csv" %DATA_DIR)
    testY = csv_to_numpy_array("%stestY.csv" %DATA_DIR)

    return trainX,trainY,testX,testY



def conv_layer(input, size_in, size_out, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([3, size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="b")
        conv = tf.nn.conv1d(input, w, stride=1, padding="SAME")
        act = tf.nn.relu(conv + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act
        #return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def fc_layer(input, size_in, size_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="b")
        act = tf.nn.softmax(tf.matmul(input, w) + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act


def dropout_layer(input, keep_probability, name="dropout"):
    with tf.name_scope(name):
        do = tf.nn.dropout(input, keep_probability)
        tf.summary.histogram("dropout", do)
        return do


def mnist_model(learning_rate, numberOfSteps, loadModel, hparam, task):
  
    trainX,trainY,testX,testY = import_data()
  
    tf.reset_default_graph()

    # Setup placeholders, and reshape the data
    x = tf.placeholder(tf.float32, shape=[None, 7], name="x")
  
    y = tf.placeholder(tf.float32, shape=[None, 2], name="labels")

    if MODEL_NUMBER == 1:

        x_reshaped = tf.reshape(x, [-1,7,1])

        conv1 = conv_layer(x_reshaped, 1, 32, name="conv1")

        flattened = tf.reshape(conv1, [-1, 7 * 32]) #=224

        do = dropout_layer(flattened, 0.5)

        logits = fc_layer(do, 224, 2, "logits")

    if MODEL_NUMBER == 2:

        x = tf.nn.l2_normalize(x, 1)

        x_reshaped = tf.reshape(x, [-1,7,1])

        conv1 = conv_layer(x_reshaped, 1, 32, name="conv1")    

        conv2 = conv_layer(conv1, 32, 128)

        flattened = tf.reshape(conv2, [-1, 7 * 128]) #=896

        fc1 = fc_layer(flattened, 896, 224, "fully_connected_layer")

        do = dropout_layer(fc1, 0.5)

        logits = fc_layer(do, 224, 2, "logits")

    if MODEL_NUMBER == 3:

        x = tf.nn.l2_normalize(x, 1)

        fc1 = fc_layer(x, 7, 512, "fully_connected_layer_1")

        if TASK == 'Train': 
            drop_per = 0.5
        else:
            drop_per = 0.9

        do = dropout_layer(fc1, drop_per)

        fc2 = fc_layer(do, 512, 224, "fully_connected_layer_2")

        logits = fc_layer(fc2, 224, 2, "logits")

    if MODEL_NUMBER == 4:

        x_reshaped = tf.reshape(x, [-1,7,1])

        conv1 = conv_layer(x_reshaped, 1, 32, name="conv1")

        flattened = tf.reshape(conv1, [-1, 7 * 32]) #=224

        do = dropout_layer(flattened, 0.3)

        logits = fc_layer(do, 224, 2, "logits")

    if MODEL_NUMBER == 5:
    
        logits = fc_layer(x, 7, 2, name="logits")


    output = tf.argmax(logits, 1)


    with tf.name_scope('xent'):
        xent = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y),
            name="x_ent")
        tf.summary.scalar('cross_entropy', xent)
  
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(xent)
        #rain_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(xent)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    sess = tf.Session()
  
    merged_summary = tf.summary.merge_all()

    writer = tf.summary.FileWriter(LOGDIR + hparam)

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    checkpointName = LOGDIR + hparam + '/checkpoints'

    if loadModel:
        print('Trying to load previous model from: %s' %(LOGDIR + hparam + 'checkpoint'))
        try: 
            f = open(LOGDIR + hparam + 'checkpoint', 'r')
            cp_path = f.readline()
            f.close()
            cp_path = cp_path[cp_path.find('"')+1 : cp_path.rfind('"')]

            saver.restore(sess, LOGDIR + hparam + cp_path)
            print('Model succesfully restored from: %s.' %(cp_path))

        except FileNotFoundError:
            print('Can not load model: no checkpoint found.')


    writer.add_graph(sess.graph)
    if task == 'Train':
        for i in range(numberOfSteps):
            X = trainX
            Y = trainY

            if i % 5 == 0:
                s = sess.run(merged_summary, feed_dict={x: X, y: Y})
                writer.add_summary(s, i)

            if i % 1000 == 0:
                drop_per = drop_per + 0.0004
                [train_accuracy, loss, out] = sess.run(
                    [accuracy, xent, output], feed_dict={x: X, y: Y})
                print("Step %d, loss %g, training accuracy %g" %(i, loss, train_accuracy))
            if i % 10000 == 0:
                print(out)

            if i % 10000 == 0 and i > 0:
                print('Saving checkpoint.')
                saver.save(sess, checkpointName, global_step=i)

            if i % 4000 == 0:
                learning_rate = learning_rate / 10

            sess.run(train_step, feed_dict={x: X, y: Y})

    if task == 'Predict':
        logitss = []
        accuracies = []
        for i in range(50):
            logitss.append(sess.run(output, feed_dict={x: testX, y:testY}))
            #print(testX, testY)
            accuracies.append(sess.run(accuracy, feed_dict={x: testX, y: testY}))
        print(logitss)
        print(accuracies)


def main():

    loadModel = LOAD_MODEL
    numberOfSteps = NUM_OF_STEPS
    learning_rate = LEARNING_RATE 

    hparam = ''

    mnist_model(learning_rate, numberOfSteps, loadModel, hparam, TASK)



if __name__ == '__main__':
    main()
