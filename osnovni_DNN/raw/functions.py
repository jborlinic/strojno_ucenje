import tensorflow as tf
import numpy as np
import pandas

def add_layer(inputs, in_size, out_size, activation_function = None, dropout = False):
    #inputs = data we get from the last layer
    #in_size = the size of the output of that layer
    #out_size = the size of the output for this layer
    #activation_function = the activation function we want to use on this layer
    print(inputs)
    Weights = tf.Variable(tf.random_normal([in_size, out_size]), 
                                           name='Weights', 
                                           dtype=tf.float32)
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,
                                  name='biases',
                                  dtype=tf.float32)

    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    if activation_function is None:
        outputs = Wx_plus_b
    elif dropout:
        outputs = activation_function(Wx_plus_b, 0.5)
    else:
        outputs = activation_function(Wx_plus_b)

    return outputs



#fills up the array with missing data (median or average)
#transforms the "importnant" coloumns to numbers (male,female -> 0,1)
#returns a sub-array of calculable data

def parse_data(d, training):

    d['Age'] = d['Age'].fillna(d['Age'] .median())
    d['Embarked'] = d['Embarked'].fillna('S')
    d['Fare'] = d['Fare'].fillna(d['Fare'].median())

    d.loc[d['Sex'] == 'male', 'Sex'] = 0
    d.loc[d['Sex'] == 'female', 'Sex'] = 1

    d.loc[d['Embarked'] == 'S', 'Embarked'] = 0
    d.loc[d['Embarked'] == 'C', 'Embarked'] = 1
    d.loc[d['Embarked'] == 'Q', 'Embarked'] = 2

#

    r1 = np.array(
        d.loc[
            :,
            ['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
        ],
        dtype=float)

    if(training):
        r2 = np.array(d.loc[:,['PassengerId', 'Survived']], dtype=float)
        return r1, r2

    return r1


#randomly selects a batch of data for the machine learning algorithm
#returns batch_xs and batch_ys

def batcher(input_data, train_results, batch_size):
    
    if(batch_size > len(input_data)):
        print('Size missmatch')
        return (Null, Null)
    
    else:
        indices = np.concatenate((
                                np.ones(batch_size),
                                np.zeros(len(input_data) - batch_size)), axis = 0)

        np.random.shuffle(indices)
        indices = (indices == True)
        batch_xs = input_data[indices]

        batch_ys = train_results[indices]
        
        #print(batch_xs, batch_ys)

        #removes first column
        
        batch_xs = batch_xs[:,np.arange(len(batch_xs[0]))[1:]]
        batch_ys = batch_ys[:,np.arange(len(batch_ys[0]))[1:]]
        
        #transforms an 1-D [0,1] array to a 2-D [[1, 0], [0, 1]] array
        batch_ys = batch_ys.T[0]
        
        batch_ys = np.array((1-batch_ys, batch_ys)).T

        #print(batch_ys)

        return  batch_xs, batch_ys

