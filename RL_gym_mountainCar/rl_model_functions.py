import keras 
from keras.layers import Flatten, Dense, Dropout
from keras.models import model_from_json

def neural_network_model(input_size):
    model = keras.models.Sequential()    
    
    model.add(Dense(128, activation='relu', input_shape=(input_size,)))
    print(model.outputs)
    model.add(Dropout(0.5))
    
    model.add(Dense(258, activation='relu'))
    print(model.outputs)
    model.add(Dropout(0.5))
    
    model.add(Dense(512, activation='relu'))
    print(model.outputs)
    model.add(Dropout(0.5))
    
    model.add(Dense(268, activation='relu'))
    print(model.outputs)
    model.add(Dropout(0.5))
    
    model.add(Dense(128, activation='relu'))
    print(model.outputs)
    model.add(Dropout(0.5))

    model.add(Dense(3, activation="softmax"))
    print(model.outputs)

    return model


def load_model(model_name='model', model_dir='model/'):
    try:
        f = open(model_dir + model_name, 'r')
        json_string = f.read()

        model = model_from_json(json_string)
        model.load_weights('%sweights' %model_dir)
        print('Model loaded!')

        return model

    except FileNotFoundError:
        print('Could not load model!')
        return None


def save_model(model, model_name='model', model_dir='model/'):
    json_string = model.to_json()

    f = open(model_dir + model_name, 'w+')
    f.write(json_string)
    f.close()

    model.save_weights('%sweights' %model_dir)
    print("Model saved!")


def train_model(train_X, train_Y, lr=1e-3, bs=100, epochs=5, model=False):
    
    if not model:
        model = neural_network_model(input_size=len(train_X[0]))

    model.compile(loss='mean_absolute_error', 
                  optimizer='RMSprop', 
                  learning_rate=lr)


    model.fit(train_X, train_Y, epochs=epochs, batch_size=bs)

    return model