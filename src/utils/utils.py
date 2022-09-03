from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import pickle


def tulosta_otsake(otsikko):
    print("*" * 70)
    print("*" + " MNIST-numerontunnistus ".center(68) + "*")
    print("*" + f" {otsikko} ".center(68) + "*")
    print("*" * 70 + "\n")


def lataa_mnist():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype("float32") / 255
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype("float32") / 255
    return (train_images, train_labels), (test_images, test_labels)


def piirra_kaaviot_tf(history):
    # summarize history for accuracy
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()
    # summarize history for loss
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()


def piirra_kaaviot_oma(historia):
    # Tee kaavio tarkkuuden kehittymisestä
    plt.plot(historia["tarkkuus"])
    plt.plot(historia["validointitarkkuus"])
    plt.title("Mallin tarkkuus")
    plt.ylabel("tarkkuus")
    plt.xlabel("epookki")
    plt.legend(["koulutus", "testi"], loc="upper left")
    plt.show()
    # Tee kaavio hukka-arvon kehittymisestä
    plt.plot(historia["hukka"])
    plt.plot(historia["validointihukka"])
    plt.title("Mallin hukka-arvo")
    plt.ylabel("hukka")
    plt.xlabel("epookki")
    plt.legend(["koulutus", "testi"], loc="upper left")
    plt.show()


def piirra_kaaviot(historia, history):
    # summarize history for accuracy
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.plot(historia["tarkkuus"])
    plt.plot(historia["validointitarkkuus"])
    plt.title("Mallin tarkkuus")
    plt.ylabel("tarkkuus")
    plt.xlabel("epookki")
    plt.legend(["tf_koulutus", "tf_testi", "oma_koulutus", "oma_testi"], loc="upper left")
    plt.show()
    # summarize history for loss
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.plot(historia["hukka"])
    plt.plot(historia["validointihukka"])
    plt.title("Mallin hukkafunktion kehitys")
    plt.ylabel("hukka")
    plt.xlabel("epookki")
    plt.legend(["tf_koulutus", "tf_testi", "oma_koulutus", "oma_testi"], loc="upper left")
    plt.show()


def lataa_malli(tiedostonimi: str) -> None:
    """Lataa neuroverkko tiedostosta"""
    with open(tiedostonimi, "rb") as f:
        verkko = pickle.load(f)
    return verkko


def tallenna_malli(verkko, tiedostonimi: str) -> None:
    """Tallenna neuroverkko tiedostoon"""
    with open(tiedostonimi, "wb") as f:
        pickle.dump(verkko, f)
