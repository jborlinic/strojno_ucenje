import tensorflow as tf
import urllib.request
import pandas as pd

LOGDIR = 'logNew/'

### MNIST EMBEDDINGS ###
mnist = tf.contrib.learn.datasets.mnist.read_data_sets(train_dir='../../.datasets/mnist/', one_hot=True)


def conv_layer(input, size_in, size_out, name="conv"):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="b")
    conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
    act = tf.nn.relu(conv + b)
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", act)
    return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def fc_layer(input, size_in, size_out, name="fc"):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="b")
    act = tf.nn.relu(tf.matmul(input, w) + b)
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", act)
    return act


def dropout_layer(input, keep_probability, name="dropout"):
  with tf.name_scope(name):
    do = tf.nn.dropout(input, keep_probability)
    tf.summary.histogram("dropout", do)
    return do


def mnist_model(learning_rate, numberOfSteps, use_two_conv, use_two_fc, hparam_str, loadModel, writeResults):
  tf.reset_default_graph()

  # Setup placeholders, and reshape the data
  x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
  x_image = tf.reshape(x, [-1, 28, 28, 1])
  
  tf.summary.image("input", x_image, 5)
  
  y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")

  if use_two_conv:
    conv1 = conv_layer(x_image, 1, 32, "conv1")
    conv_output = conv_layer(conv1, 32, 64, "conv2")
    

  else:
    conv1 = conv_layer(x_image, 1, 64, "conv1")
    conv_output = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    
  conv_image = tf.reshape(tf.slice(conv_output, [0,0,0,0], [1,7,7,64]), [64,7,7,1])

  tf.summary.image("conv_output", conv_image, 64)

  flattened = tf.reshape(conv_output, [-1, 7 * 7 * 64])

  if use_two_fc:
    fc1 = fc_layer(flattened, 7 * 7 * 64, 1024, "fc1")
    do = dropout_layer(fc1, 0.5)
    logits = fc_layer(do, 1024, 10, "fc2")
    

  else:
    logits = fc_layer(flattened, 7 * 7 * 64, 10, "fc1")

  output = tf.argmax(logits, 1)

  with tf.name_scope('xent'):
    xent = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y), name="x_ent")
    tf.summary.scalar('cross_entropy', xent)
  
  with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(xent)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)
  

  sess = tf.Session()
  
  merged_summary = tf.summary.merge_all()

  writer = tf.summary.FileWriter(LOGDIR + hparam_str)

  sess.run(tf.global_variables_initializer())

  saver = tf.train.Saver()
  checkpointName = LOGDIR + hparam_str + '/myMNIST_Model'

  if loadModel:
    print('Trying to load previous model from: %s' %(LOGDIR + hparam_str + '/'))
    try: 
      f = open(LOGDIR + hparam_str + '/checkpoint', 'r')
      cp_path = f.readline()
      f.close()
      cp_path = cp_path[cp_path.find('"')+1 : cp_path.rfind('"')]

      saver.restore(sess, cp_path)
      print('Model succesfully restored from: %s.' %(cp_path))

    except FileNotFoundError:
      print('Can not load model: no checkpoint found.')


  writer.add_graph(sess.graph)

  for i in range(numberOfSteps):
    batch = mnist.train.next_batch(100)
    if i % 5 == 0:
      s = sess.run(merged_summary, feed_dict={x: batch[0], y: batch[1]})
      writer.add_summary(s, i)

    if i % 500 == 0:
      [train_accuracy] = sess.run([accuracy], feed_dict={x: batch[0], y: batch[1]})
      print("Step %d, training accuracy %g" %(i, train_accuracy))

    if i % 10000 == 0 and i > 0:
      print('Saving checkpoint.')
      saver.save(sess, checkpointName, global_step=i)

    if i % 4000 == 0:
      learning_rate = learning_rate / 10

    sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})


  if writeResults:
    testData = pd.read_csv(LOGDIR + 'data/test.csv')

    outputData = pd.Series()
    
    for i in range(280):
      outputPart = pd.Series(sess.run(output, feed_dict={x: testData[i*100 : i*100 + 100]}))
      
      outputData = outputData.append(outputPart, ignore_index=True)

    outputData.index = outputData.index + 1 #indexes start with 1
    outputData.name = 'Label'
    outputData.to_csv(LOGDIR + hparam_str + '/output.csv', index_label='ImageId', header=True)
    print('Output saved.')



def setLogDir(newRun):
    try: 
      f = open(LOGDIR + 'runNumber', 'r')
      runNumber = f.read();

      if newRun:
        runNumber = str(int(runNumber) + 1)
      
        f.close()
        f = open(LOGDIR + 'runNumber', 'w')
      
        f.write(runNumber)
      
      f.close()

      return runNumber + '/'

    except FileNotFoundError:
      f = open(LOGDIR + 'runNumber', 'w')
      
      f.write('0');
      
      f.close()

      return setLogDir(newRun)
  


def make_hparam_strin(learning_rate, use_two_fc, use_two_conv, runNumber):
  fc = 1
  conv = 1
  if use_two_conv:
    conv += 1

  if use_two_fc:
    fc += 1

  return '%slr_%.0E__fc_%d__conv_%d' %(runNumber, learning_rate, fc, conv)



def main():
  
  loadModel = False
  
  writeResults = True

  runNumber = setLogDir(not loadModel)

  numberOfSteps = 5001
  learning_rates = [1E-3, 1E-4, 1E-5]
  two_fc = [True, False]
  two_conv = [True, False]



  for learning_rate in learning_rates:
    for use_two_fc in two_fc:
      for use_two_conv in two_conv:
        hparam_str = make_hparam_strin(learning_rate, use_two_fc, use_two_conv, runNumber)

        mnist_model(learning_rate, numberOfSteps, use_two_conv, use_two_fc, hparam_str, loadModel, writeResults)


if __name__ == '__main__':
  main()
