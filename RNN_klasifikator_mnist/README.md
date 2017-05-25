### Primerjava orodij za implementacijo RNN-modelov
Ponavljajoče nevronske mreže imajo težjo implementacijo in zato za njihovo implementacijo obstaja dosti več ogrodij, kot za splošne nevronske mreže.  
Tukaj sem se odločil preizkusiti tri ogrodja, ki temeljijo na knjižnici Tensorflow.

Kljub temu, da so RNN-ji bolj priljubljeni za delo s časovno odvisnimi nabori podatkov, jih lahko uporabimo tudi na drugih oblikah podatkov. Iskal sem nabor podatkov, ki bi bil dobro dokumentiran, lahko dostopen in preprost za razumevanje. Ob pregledu vseh ogrodij je edini nabor podatkov, ki ima v vseh treh ogrodjih lahko implementacijo, je enostaven in ga tudi že poznamo je bil nabor podatkov ročno napisanih števil [MNIST](http://yann.lecun.com/exdb/mnist/).

#### Knjižnica Tensorflow [povezava](https://github.com/jborlinic/machine_learning/blob/master/RNN_klasifikator_mnist/RNN_classificator_MNIST_basic.ipynb)
Glede na to, da vsa orodja temeljijo na tej knjižnici sem za osnovno implementacijo želel uporabiti le knjižnico Tensorflow. 
Zelo lep primer sem našel na [youtubu](https://www.youtube.com/watch?v=SeffmcG42SY&index=20&list=PLXO45tsB95cJHXaDKpbwr5fC_CCYylw1f). Je zelo lepo napisana osnovna koda, ki implementira RNN model s [LSTM celico](http://colah.github.io/posts/2015-08-Understanding-LSTMs/). LSTM celica je modificirana RNN celica, ki se z dodatnimi verjetnostnimi vrati izogne neskončnemu gradientu in omogoča dolg kratkotrajen spomin.  

#### Implementacija s pomočjo contrib.rnn znotraj knjižnice Tensorflow [povezava](https://github.com/jborlinic/machine_learning/blob/master/RNN_klasifikator_mnist/RNN_classificator_MNIST_TF_contrib_rnn.ipynb)
Knjižnica Tensorflow je odprtokodna, kar pomeni, da lahko k njej doprinašajo tudi ne-Googlovi razvijalci. Ker pa je vseeno podprta s strani Googla, pa upoštevajo zunanje posodobitve preko posebnega sistema preverjanja kvalitete kode in s tem zagotavljanja stabilnosti.  
Proces pridobivanja nove kode je dolgotrajen, saj mora za njo nastati dovolj velika potreba, nekdo jo mora kvalitetno napisati in nato mora prenesti masovno testiranje. Ta proces deluje preko podknjižnice contrib in znotraj nje najdemu tudi implementacijo statičnega RNN modela.  
Ta primer uporablja to implementacijo.

#### Implementacija s pomočjo contrib.keras znotraj knjižnice Tensorflow [povezava](https://github.com/jborlinic/machine_learning/blob/master/RNN_klasifikator_mnist/RNN_classificator_MNIST_TF_contrib_keras.ipynb)
V podknjižnici contrib najdemo tudi delno implementacijo visoko-nivojskega API-ja Keras. Keras je knjižnica, ki poenostavi delo s nevronskimi mrežami, ponuja obstoječe implementacije posameznih slojev in upošteva najboljše implementacije posameznih nivojev. Je zelo dostopna in odlična za začetnike ter strokovnjake, ki želijo na hitro sprobati več različnih modelov. Keras za osnovo uporablja python in knjižnico Tensorflow ali knjižnico Theano, pred kratkim pa so se združili s ekipo Tensorflowa in sedaj želijo knjižnici združiti in ustvariti visoko-nivojsko plast knjižnice Tensorflow. Ta implementacija je dolgotrajna, je pa večina osnovnih modelov, vključno s RNN-modelom, že implementirana in jo najdemo v podknjižnici contrib.keras.  
Ta primer uporablja to implementacijo.
