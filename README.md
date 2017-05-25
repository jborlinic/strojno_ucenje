# strojno učenje - *machine learning*


To je repozitorij v katerem shranjujem kodo in "tutoriale", ki sem jih izdelal tekom svojega učenja tega področja.

Uporabljam:
- OS: Ubuntu 16.04
- Programski jezik: Python
- Knjižnice:
    - Tensorflow-gpu
    - Keras
    - gym

## Obstoječi modeli
### Globoke nevronske mreže - DNN (Deep Neural Network)

#### Preživetje na Titaniku [povezava](https://github.com/jborlinic/machine_learning/blob/master/osnovni_DNN/DNN-titanik.ipynb)  
_Kratek opis:_  
Osnovna nevronska mreža za napovedovanje preživelih oseb na titaniku. Nabor podatkov pridobljen na spletni platformi [Kaggle](https://www.kaggle.com/c/titanic), kjer je to eno izmed začetniških tekmovanj.  
Vsebuje opis in pokomentirano kodo.

### Konvolucijske nevronske mreže - CNN (Convolutional Neural Network)

#### Klasifikacija ročno napisanih števil [povezava](https://github.com/jborlinic/machine_learning/blob/master/CNN_konvolucijske_mreze/CNN_s_Tensorflow.ipynb)
_Kratek opis:_  
Začetni primer konvolucijske nevronske mreže in vuzalizacije ter nadzorovanja procesa učenja s pomočjo orodja Tensorboard. Klasifikacija se izvaja na najbolj prepoznavnem naboru podatkov za klasifikacijo slik, [MNIST handwritten digits](http://yann.lecun.com/exdb/mnist/).  
Vsebuje opis in pokomentirano kodo.

#### Klasifikacija slik v 10 razredov [povezava](https://github.com/jborlinic/machine_learning/blob/master/CNN_klasifikator-barvne_slike/Konvolucijski_klasifikator_cifar-10.ipynb)
_Kratek opis:_  
Nadgradnja zgornjega modela. Prehod iz 2-D konvolucije na 3-D konvolucijo s pomočjo nabora podatkov [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).  
V direktoriju najdemo dva modela: 2-D model uporablja črno-bele slike, 3-D model pa originalne barvne slike.  
Vsebuje pokomentirano kodo.

#### Alternativni model klasifikacije s knjižnico Keras [povezava](https://github.com/jborlinic/machine_learning/blob/master/CNN_mnist_keras/konvolucijska_mreza_keras.ipynb)
_Kratek opis:_  
Klasifikator ročno zapisanih števil nabora podatkov MNIST z uporabo visoko nivojske knjižnice Keras. Knjižnica drastično poenostavi implementacijo modela z upoštevanjem dobrih praks, a hkrati omogoča kasnejše fine nastavitve.  
Vsebuje opis in pokomentirano kodo.

### Ponavljajoče nevronske mreže - RNN (Recurrent Neural Network)

#### Klasifikator sintetičnih časovno odvisnih podatkov [povezava](https://github.com/jborlinic/machine_learning/blob/master/RNN_sinteticni_podatki/SinteticRNN_p1.ipynb)
_Kraterk opis:_  
Ponavljajoče nevronske mreže dosegajo najboljše natančnosti pri časnovno odvisnih podatkih, saj so zgrajene tako, da imitirajo dolg kratkotrajen spomin.  
To je dvodelni začetni primer takšne nevronske mreže, ki se uči na sintetičnih podatkih.  
Vsebuje pokomentirano kodo.

#### Klasifikator ročno napisanih števil [povezava](https://github.com/jborlinic/machine_learning/tree/master/RNN_klasifikator_mnist)
_Kraterk opis:_  
Štiri različne implementacije modelov ponavljajoče nevronske mreže na že znanem naboru podatkov [MNIST](http://yann.lecun.com/exdb/mnist/).  
Implementacije se med seboj razlikujejo po uporabi različnih (delov) knjižnic Tensorflow in Keras. Tukaj sem iskal najboljši način za implementacijo modelov.  
Preizkusil sem:
    - Osnovni Tensorflow,
    - uporabo Tensorflow contrib rnn (tf.contrib.rnn),  
    - uporabo Tensorflow contrib keras (tf.contrib.keras)
Trenutno se mi zdi najbolj uporabna knjižnica Keras z jedrom Tensorflow, kmalu bo knjižnica tf.contrib.keras (kasneje le tf.keras) zamenjala uporabo knjižnice keras a še nismo tam.  
Vsebuje opis in pokomentirano kodo.

#### Klasifikator filmskih opisov [povezava](https://github.com/jborlinic/machine_learning/tree/master/RNN_klasifikator_IMDB)
_Kratek opis:_  
Knjižnica Keras ima nekaj že implementiranih naborov podatkov (npr. MNIST, CIFAR-10, CIFAR-100), za njih najdemo vse potrebne funkcije za pred-procesiranje podatkov, kar pospeši učni proces.  Med njimi je tudi nabor podatkov filmskih opisov spletne strani [IMDB](http://www.imdb.com). Ti so že vektorizirani. Kot oznake so klasificirani med pozitivne in negativne.  
S tem naborom podatkov je naučen model ponavljajoče nevronske mreže, ki oceni nek vektoriziran filmski opis ali je pozitiven ali negativen.  
Vsebuje opis in pokomentirano kodo.

#### Napovedovanje cene Bitcoinov [povezava](https://github.com/jborlinic/machine_learning/tree/master/RNN_bitcoin)
_Kratek opis:_  
Ko razmišljamo o časnovno odvisnih naborih podatkov hitro pridemo do cene delnic ali spletnih valut. Tukaj sem s pomočjo ponavljajoče nevronske mreže poizkušal napovedati vrednost valute Bitcoin. Model še potrebuje dodatne nastavitve za boljšo natančnost in še ni uporaben za trgovanje.  
Zraven visokega cilja, napovedati ceno Bitcoina, je model nastevlnjen tako, da ga lahko testiramo na preprostejših ciljih (npr. napovedati vrednosti zaporedja števil, funkcijo sinus).  
Vsebuje opis in pokomentirano kodo.

### Samodejn kodirnik - AE (AutoEncoder)
#### Samodejni kodirnik za raočno zapisana števila [povezava](https://github.com/jborlinic/machine_learning/tree/master/AE_mnist)
_Kratek opis:_  
Samodejn kodirnik (_ang. autoencoder_) je tip nevronske mreže, ki se uporablja za ustvarjanje slik, glasbe in za samodejno klasifikacijo nabora podatkov. Sestavljen je iz dveh delov, kodirnega in dekodirnega dela, ki pa sta enaka, le da je zaporedje slojev obrnjeno. Tako lahko uporabimo vse, do sedaj znane, nevronske mreže za njegovo implementacijo.  
Kot primer, sem implementiral dva samodejna kodirnika, enega na osnovi navadne nevroneke mreže in drugega na osnovi konvolucijske nevronske mreže.  
Ker pa je prvi primer nenadzorovanega učenja uporabim, že dobro znan naboru podatkov [MNIST](http://yann.lecun.com/exdb/mnist/). Zaenkrat še ima model težave z neenakomerno porazdelitvijo slik v razrede, se pa nauči ločiti določene oblike, (npr. ukrivljan črta -št. 0, 2, 6, 8, 9, ravna črta št. 1,4,7).  
Vsebuje opis in pokomentirano kodo.

## TO-DO:

- Dokončati STS sintetic (sintetični primer sequence to sequence modela - kvadratne funkcije, polinomi, kvadratne funkcije iz poljubne strani, polinomi iz poljubne strani).
- Poizkati dober nabor podatkov za implementacijo STS modela.
- Implementirati STS model na zgoraj pridobljenem naboru podatkov.
- Implementirati SIS model na naboru podatkov GPS-tracks (napovedovanje zaporedja naslednjih gps točk).
- Implementacija Q-learning modela za reševanje diskretnih RL nalog.
    -  primer Open AI gym [CartPole-v0](https://gym.openai.com/envs)
- Implementacija DDPG (deep deterministic policy gradient) odela za reševanje zveznih RL nalog.
    -  primer Open AI gym [MountainCarContinuous-v0](https://gym.openai.com/envs)
