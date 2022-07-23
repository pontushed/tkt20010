# Määrittelydokumentti

Käytetty ohjelmointikieli: **Python**

Hallitsen myös seuraavat kielet: Java, JavaScript

_Mitä algoritmeja ja tietorakenteita toteutat työssäsi?_

Neuroverkko

_Mitä ongelmaa ratkaiset ja miksi valitsit kyseiset algoritmit/tietorakenteet?_

Numerontunnistus MNIST-tietokannalla koulutetulla neuroverkolla. Haluan toteuttaa neuroverkon käytännössä ja verrata sitä valmiiseen toteutukseen (TensorFlow/Keras)

_Mitä syötteitä ohjelma saa ja miten näitä käytetään?_

Neuroverkko koulutetaan MNIST-tietokannalla.
Ohjelmalle syötetään käsinkirjoitettu numero. Ohjelman tarkoitus on päätellä kuvan perusteella, mikä numero on kyseessä (0-9).

_Tavoitteena olevat aika- ja tilavaativuudet (m.m. O-analyysit)_

_Lähteet_

_Muut tiedot_

Opinto-ohjelma: **TKT**

Projektissa käytetty kieli: **Suomi**

## Referenssitaso

Olen luonut vertailua varten referenssitoteutuksen käyttäen TensorFlow/Keras-kirjastoa. Siinä neuroverkon arkkitehtuuri on:

- yksi piilotettu taso 512 neuronilla, aktivointifunktio ReLU
- ulostulotaso 10 neuronilla, aktivointifunktio SoftMax

Ajo:

```shell
$ poetry install
$ poetry shell
$ python src/referenssitaso.py
```

Tulostus:

```
**********************************************************************
*                       MNIST-numerontunnistus                       *
*          Referenssitoteutus TensorFlow/Keras-kirjastolla           *
**********************************************************************

TensorFlow version: 2.9.1
[1/4] Ladataan MNIST-datasetti...
[2/4] Määritellään neuroverkko...
2022-07-23 10:58:57.572110: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
[3/4] Koulutetaan neuroverkkoa...
Epoch 1/5
469/469 [==============================] - 2s 4ms/step - loss: 0.2562 - accuracy: 0.9257
Epoch 2/5
469/469 [==============================] - 2s 4ms/step - loss: 0.1037 - accuracy: 0.9689
Epoch 3/5
469/469 [==============================] - 2s 4ms/step - loss: 0.0687 - accuracy: 0.9794
Epoch 4/5
469/469 [==============================] - 2s 4ms/step - loss: 0.0503 - accuracy: 0.9853
Epoch 5/5
469/469 [==============================] - 2s 4ms/step - loss: 0.0375 - accuracy: 0.9890
[4/4] Testataan neuroverkkoa...
1/1 [==============================] - 0s 61ms/step
313/313 [==============================] - 1s 2ms/step - loss: 0.0681 - accuracy: 0.9803

Neuroverkon tarkkuus testidatalla: 98.03%
```
