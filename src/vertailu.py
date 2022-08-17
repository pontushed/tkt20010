# Vertailu referenssitason ja oman toteutuksen välillä
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import Sequential, layers
from time import perf_counter
import matplotlib.pyplot as plt
from neuroverkko import Neuroverkko, Tihea, ReLU

print("*" * 70)
print("*" + " MNIST-numerontunnistus ".center(68) + "*")
print("*" + " Vertailu oman toteutuksen ja TensorFlown välillä ".center(68) + "*")
print("*" * 70 + "\n")

print("TensorFlow versio:", tf.__version__)

# Parametrit
epochs = 10
batch_size = 100

# Ladataan MNIST-datasetti
print("[1/4] Ladataan MNIST-datasetti...")
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

# Neuroverkkojen määrittely
print("[2/4] Määritellään neuroverkot...")
model = Sequential(
    [
        layers.Dense(512, input_shape=(784,), activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)
VERKKO = [Tihea(train_images.shape[1], 512), ReLU(), Tihea(512, 10)]

# Neuroverkon rakennus
model.compile(optimizer="RMSprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
neuroverkko = Neuroverkko(VERKKO)


# Neuroverkon kouluttaminen
print("[3/4] Koulutetaan neuroverkot...")
start = perf_counter()
history = model.fit(
    train_images,
    train_labels,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(test_images, test_labels),
)
end = perf_counter()
print(f"Tensorflown koulutus kesti {end - start:.3f} sekuntia.")
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"\nTF-Neuroverkon tarkkuus testidatalla (n={len(test_images)}): {test_acc*100:.2f}%")

start = perf_counter()
historia = neuroverkko.sovita(
    train_images,
    train_labels,
    epookit=epochs,
    alijoukon_koko=batch_size,
    validaatiodata=(test_images, test_labels),
)
end = perf_counter()
print(f"Oman neuroverkon koulutus kesti {end - start:.5f} sekuntia.")
tarkkuus = neuroverkko.evaluoi(test_images, test_labels)
print(f"\nOman neuroverkon tarkkuus testidatalla (n={len(test_images)}): {tarkkuus*100:.2f}%")

# Tulostetaan yhteenveto
print("[4/4] Tulostetaan yhteenveto...")
# summarize history for accuracy
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.plot(historia["tarkkuus"])
plt.plot(historia["validointitarkkuus"])
plt.title("Mallin tarkkuus")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["tf_koulutus", "tf_testi", "oma_koulutus", "oma_testi"], loc="upper left")
plt.show()
# summarize history for loss
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.plot(historia["hukka"])
plt.plot(historia["validointihukka"])
plt.title("Mallin hukkafunktion kehitys")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["tf_koulutus", "tf_testi", "oma_koulutus", "oma_testi"], loc="upper left")
plt.show()
