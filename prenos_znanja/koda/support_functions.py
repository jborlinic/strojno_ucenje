import numpy as np
from keras.models import model_from_json


def one_hot(obj, n_classes):
    """
    Ta metoda preoblikuje oznako (0 ... 9) v "one hot" vektor velikosti (st_različnih_oznak, 1) 
    Primer: 5 -> [0,0,0,0,0,1,0,0,0,0].
    
    Metoda je napisana za pretvorbo numpy matrik.
    [1,     [[0,1,0],
     0, ->   [1,0,0],
     2]      [0,0,1]]
    
    """
    obj_len = len(obj)
    a = np.array(obj)
    b = np.zeros((obj_len, n_classes))
    b[np.arange(obj_len), a] = 1

    return b


def save_model(model_dir, model):
    """
    Ta metoda shrani naucen model v mapo: model_dir.
    """
    json_string = model.to_json()

    f = open('%smodel' %model_dir, 'w+')
    f.write(json_string)
    f.close()

    model.save_weights('%sweights' %model_dir)
    print("Model saved!")


def load_model(model_dir):
    """
    Ta metoda nalozi obstojec model (shranjen z metodo save_model zgoraj) iz mape model_dir.

    """
    try:
        f = open('%smodel' %model_dir, 'r')
        json_string = f.read()

        model = model_from_json(json_string)
        model.load_weights('%sweights' %model_dir)
        print('Model loaded!')

        return model

    except FileNotFoundError:
        print('Could not load model!')