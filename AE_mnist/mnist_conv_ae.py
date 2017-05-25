import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras.models import model_from_json
from keras.datasets import mnist
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D,\
                         UpSampling2D, Dense, Reshape, Dropout

def toOneHot(array, n_classes):
    # Transforms an array of labels (dim: (-1,)) to a 1-hot array of labels (dim: (-1, n_classes))
    temp_arr = np.zeros((len(array), n_classes))
    temp_arr[np.arange(len(array)), array] = 1
    
    return temp_arr

def getLayerOutput(model, data=None):
    # Function that returns the outputs of each layer in the model.
    from keras import backend as K

    inp = model.input                                           # input placeholder
    outputs = [layer.output for layer in model.layers]          # all layer outputs
    functor = K.function([inp]+ [K.learning_phase()], outputs ) # evaluation function
    
    if data is None:
        data = np.random.rand(model.input_shape)

    layer_outs = functor([data, 1])
    
    return layer_outs

def load_data():
    # Loads and normalizes the MNIST dataset.
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    return (x_train, y_train), (x_test, y_test)

def build_model():
    # Builds (and prints the shape of) the model.
    
    model = models.Sequential()

    ### Encoder
    model.add(Reshape((28,28,1), input_shape=(28, 28))) #0
    print(model.output.shape)
    model.add(Conv2D(16, 
                     kernel_size=3, 
                     strides=1, 
                     padding="same", 
                     activation='relu'))                #1

    model.add(MaxPooling2D((2,2), padding="same"))      #2
    print(model.output.shape)
    model.add(Conv2D(32, 
                     kernel_size=3, 
                     strides=1, 
                     padding="same", 
                     activation='relu'))                #3
    print(model.output.shape)
    model.add(MaxPooling2D((2,2), padding="same"))      #4
    print(model.output.shape)
    model.add(Reshape((7*7*32,)))                       #5

    model.add(Dropout(0.7))                             #6
    print(model.output.shape)
    model.add(Dense(40, activation="softmax"))          #7
    print(model.output.shape)

    ### Decoder

    model.add(Dense(7*7*32))                            #8
    print(model.output.shape)
    model.add(Reshape((7,7,32)))                        #9
    print(model.output.shape)
    model.add(UpSampling2D(size=2))                     #10
    print(model.output.shape)
    model.add(Conv2DTranspose(16, 
                              kernel_size=3, 
                              strides=1, 
                              padding="same", 
                              activation='relu'))       #11

    model.add(UpSampling2D(size=2))                     #12
    print(model.output.shape)
    model.add(Conv2DTranspose(1, 
                              kernel_size=3, 
                              strides=1, 
                              padding="same", 
                              activation='relu'))       #13
    print(model.output.shape)
    model.add(Reshape((28, 28)))                        #14
    print(model.output.shape)

    return model


if __name__ == '__main__':
    model_dir = 'conv_model/'


    (x_train, y_train), (x_test, y_test) = load_data()
    
    # Load or build the model:
    try:

        f = open(model_dir + 'model', 'r')
        json_string = f.read()

        model = model_from_json(json_string)
        model.load_weights(model_dir + 'weights')
        print('Model loaded!')

    except FileNotFoundError:
        model = build_model()
        print('Could not load model!')

    # Initialize the model:
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')


    # Train the model:
    model.fit(x_train, 
              x_train,
              verbose=1,
              epochs=50,
              batch_size=128,
              callbacks=[TensorBoard(log_dir='log/',
                                     histogram_freq=100,
                                     write_graph=True,
                                     write_images=False)]
             )
    # Evaluate the model:
    scores = model.evaluate(x_test, x_test, verbose=0)

    print(scores)
    
    # Save the model:
    json_string = model.to_json()

    f = open(model_dir + 'model', 'w+')
    f.write(json_string)
    f.close()

    model.save_weights(model_dir + 'weights')
    print("Model saved!")

    # External model testing and evaluating, including learned feature evaluation:
    
    #x_test = x_test #toOneHot(y_test[0:500], 10)
    #x_pred = np.array(getLayerOutput(model, data=x_test)[7])
    layer_outputs = getLayerOutput(model, data=x_test)

    pred_class = np.argmax(np.array(layer_outputs[7]), axis=1)
    decoded_img = np.array(layer_outputs[-1])
    original_img = x_test
    original_class = y_test

    print(pred_class.shape)
    print(decoded_img.shape)
    print(original_img.shape)
    print(original_class.shape)


    plt.figure(figsize=(60, 2))
    for i in range(10):
        mask = pred_class == i
        plt.subplot(1, 10, i + 1)
        selected_imgs = original_img[mask]
        print('class: %d, n_imgs: %d' %(i, len(selected_imgs)))
        plt.imshow(np.average(selected_imgs, axis=0))

        plt.show()


    """


x_pred = np.argmax(x_pred, axis=1)

n = 10
mask = x_pred == 1
x_test = x_test[mask]
x_pred = x_pred[mask]
print(x_pred)
x_test = x_test[0 : 2 * n]


plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(x_test[i + n])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()"""
