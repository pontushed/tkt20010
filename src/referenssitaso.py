# Referenssitoteutus käyttäen valmista kirjastoa (TensorFlow/Keras)
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import Sequential, layers
from time import perf_counter

print("*" * 70)
print("*" + " MNIST-numerontunnistus ".center(68) + "*")
print("*" + " Referenssitoteutus TensorFlow/Keras-kirjastolla ".center(68) + "*")
print("*" * 70 + "\n")

print("TensorFlow version:", tf.__version__)

# Ladataan MNIST-datasetti
print("[1/4] Ladataan MNIST-datasetti...")
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

# Neuroverkon määrittely
print("[2/4] Määritellään neuroverkko...")
model = Sequential(
    [
        layers.Dense(512, input_shape=(784,), activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)

# Neuroverkon rakennus
model.compile(
    optimizer="RMSprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Neuroverkon kouluttaminen
print("[3/4] Koulutetaan neuroverkkoa...")
start = perf_counter()
model.fit(train_images, train_labels, epochs=1, batch_size=128)
end = perf_counter()
print(f"Koulutus kesti {end - start:.2f} sekuntia.")


# Neuroverkon testaus
print("[4/4] Testataan neuroverkkoa...")
test_digits = test_images[0:10]
predictions = model.predict(test_digits)
assert predictions[0].argmax() == test_labels[0]  # 7
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(
    f"\nNeuroverkon tarkkuus testidatalla (n={test_labels.size}): {test_acc*100:.2f}%"
)
