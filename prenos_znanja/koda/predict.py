import keras
import numpy as np

from keras.models import model_from_json
from scipy import misc

from support_functions import load_model

#####################################################################################
### Spremeni ta parameter
 
image_path = 'export2.png'

#####################################################################################
# direktorij v katerem je shranjen naš model
MODEL_DIR = 'model/'

# metoda iz direktorija MODEL_DIR naloži model in naučene uteži
model = load_model(MODEL_DIR)

if not model:
    print("V direktoriju %s ni modela." %MODEL_DIR)

else: 
    # s pomočjo metode imread preberemo sliko v matriko 
    # flatten=True nam preoblikuje RGB/CMYK v Grayscale -torej črno belo
    image = misc.imread(image_path, flatten=True)

    # slika je oblike (28,28), z np.reshape jo preoblikujemo v obliko, ki jo model uporablja
    image = np.reshape(image, (1,28,28))
    
    # z metodo predict napovemo, katero število je narisano
    print(np.argmax(model.predict(image)))