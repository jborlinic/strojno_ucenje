### Napovedovanje vrednosti BitCoinov

Ko spoznavaš strojno učenje in vse možne uporabe modelov, ki jih opisuje, hitro prideš do želje po napovedovanju cene delnic. Podobna naloga je seveda napovedovanje vrednosti valute Bitcoin.  
To idejo sem dobil tudi sam in se odločil preizkusiti, kako uspešen je lahko tak model. Glede na to, da so ponavljajoče nevronske mreže najbolj uporabljene za zaporedno odvisne nabore podatkov, so najprimernejše za napovedovanje vrednosti valute, če upoštevamo njeno dosedanje nihanje. Takšnim modelom, ko imajo kot množico rezultatov realna števila pravimo regresijski modeli. To je prvi tak model, ki sem ga napisal in je na preprostih nalogah zelo učinkovit. Na žalost, napovedovanje vrednosti valute Bitcoin __ni__ preprosta naloga. 

Nastal je RNN model, ki prejme zaporedje desetih vrednosti valute Bitcoin in vrne naslednjo predvideno vrednost.  
Model se na začetku ni učil po pričakovanjih in zato sem se odločil sprobati idejo na preprostejših nalogah z enako obliko. Te sto vključevale napovedovanje vrednosti funkcije sinus, zaporednih števil, ipd.  
Te implementacije so še v kodi in parametre se določa v datoteki *main.py*, model pa je implementiran v datoteki *ep_keras_rnn.py*.