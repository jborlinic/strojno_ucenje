import tensorflow as tf
import numpy as np
import pandas

import time
import datetime

from functions import parse_data, batcher, add_layer

#initial constants
train_file = '../../.datasets/titanic/train.csv'
test_file = '../../.datasets/titanic/test.csv'
log_path = 'log/'


#read training data
training_data = pandas.read_csv(train_file)
#print(training_data.describe())
#print(training_data)

#parse data
X, Y = parse_data(training_data, True)

#create the model

##variables
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=[None, 7], name='x_input')
    y_ = tf.placeholder(tf.float32, shape=[None, 2], name='y_input')

    #inv_matrix = tf.constant([[0,1],[1,0]], dtype = tf.float32)

##layers
with tf.name_scope('layer_0'):
     l0 = add_layer(x,7,50, activation_function=None)


# with tf.name_scope('layer_1'):
#     l1 = add_layer(x, 7, 49, activation_function = tf.nn.relu)


# with tf.name_scope('layer_2'):
#     l2 = add_layer(l1, 49, 49, activation_function = tf.nn.dropout, dropout=True)

#with tf.name_scope('layer_1'):

l0 = add_layer(x, 7, 50, activation_function = None)
l1 = add_layer(l0, 50, 300, activation_function = tf.sigmoid)
l2 = add_layer(l1, 300, 300, activation_function = tf.nn.softmax)

l4_d = add_layer(l1, 300, 300, activation_function = tf.nn.dropout, dropout=True)


l9 = add_layer(l4_d, 300, 10, activation_function = tf.sigmoid)
output = add_layer(l9, 10, 2, activation_function = tf.nn.softmax)

    
##loss
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, y_))
    tf.scalar_summary('loss', loss)




##train step
rate = 0.00001

with tf.name_scope('train_step'):
    train_step = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss)
    

##session
sess = tf.InteractiveSession()

#initialize the variables
sess.run(tf.initialize_all_variables())

#tensorboard

file_writer = tf.train.SummaryWriter(log_path, graph=sess.graph)

merged = tf.merge_all_summaries()

change_counter = 0
temp_loss = 0

#train
for i in range(3000):
#losss = 1
#i = 0
#while losss > 0.20:
    batch_xs, batch_ys = batcher(X, Y, len(X)/2)
    train_step.run(feed_dict={x: batch_xs, y_: batch_ys})
    losss = sess.run(loss, feed_dict={x:batch_xs, y_: batch_ys})
    if i % 50 == 0:
        results = sess.run(merged, feed_dict={x: batch_xs, y_: batch_ys})
        file_writer.add_summary(results, i)
    if i % 1000 == 0:
        ts = time.time()
        ts = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        #rate = rate * 0.9
        print('Step: %s, at time:' + ts) %i
        #print('Training rate: %s') % rate
        print('temp_loss: %s, repeat: %s') %(temp_loss, change_counter)
        #print(loss.eval())
        print('Loss: %s') % losss
        if temp_loss == losss:
            change_counter += 1
        
            if change_counter == 5:
                break
        else:
            change_counter = 0

        temp_loss = losss




#evaluate
test_data = pandas.read_csv(test_file)

ids = test_data['PassengerId']

test_data = parse_data(test_data, False)

test_data, temp = batcher(test_data, test_data, len(test_data))

calc = sess.run(output, feed_dict={x:test_data})

calc = tf.argmax(calc,1)

calc = np.array((ids, calc.eval()), dtype=int)

calc = calc.T

result = pandas.DataFrame(calc, columns=['PassengerId','Survived'])

result.to_csv('result.csv', index=False)
print('Results writen!')
#close session
sess.close()

#new
#------------------------------------------------------------------------------
#old


# ##################################TEST DATA:
# result = test_data[:,0]
# test_data = test_data[:,np.arange(len(test_data[0]))[1:]]

# calculated = tf.matmul(tf.to_float(test_data), W) + b

# calculated = tf.argmax(calculated,1)
# result = np.array((result, calculated.eval()), dtype = int)
# result = result.T

# result = pandas.DataFrame(result, columns=['PassengerId','Survived'])
# result.to_csv('result.csv', index=False)

# print('Results written to file: "result.csv"')
# #print(calculated.eval())
# #acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# #print(acc.eval())
