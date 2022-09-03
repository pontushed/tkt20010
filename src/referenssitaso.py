# Referenssitoteutus käyttäen valmista kirjastoa (TensorFlow/Keras)
from tensorflow.keras import Sequential, layers
from time import perf_counter
import utils.utils as utils

utils.tulosta_otsake("Referenssitoteutus TensorFlow/Keras-kirjastolla")

# Ladataan MNIST-datasetti
print("[1/4] Ladataan MNIST-datasetti...")
(X_koulutus, y_koulutus), (X_testi, y_testi) = utils.lataa_mnist()

# Neuroverkon määrittely
print("[2/4] Määritellään neuroverkko...")
malli = Sequential(
    [
        layers.Dense(512, input_shape=(784,), activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)

# Neuroverkon rakennus
malli.compile(optimizer="RMSprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
print(malli.summary())

# Neuroverkon kouluttaminen
print("[3/4] Koulutetaan neuroverkkoa...")
alku = perf_counter()
historia = malli.fit(
    X_koulutus,
    y_koulutus,
    epochs=5,
    batch_size=100,
    validation_data=(X_testi, y_testi),
)
loppu = perf_counter()
print(f"Koulutus kesti {loppu - alku:.3f} sekuntia.")


# Neuroverkon testaus
print("[4/4] Testataan neuroverkkoa...")
_, testitarkkuus = malli.evaluate(X_testi, y_testi)
print(f"\nNeuroverkon tarkkuus testidatalla (n={y_testi.size}): {testitarkkuus*100:.2f}%")

utils.piirra_kaaviot_tf(historia)
