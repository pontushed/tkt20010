from typing import Dict, Iterator, List, Tuple
import numpy as np
from numpy.typing import NDArray

from tqdm import trange  # Näyttää hienon edistysmittarin

np.random.seed(99)


class Kerros:
    """Abstrakti luokka neuroverkon kerrokselle"""

    def __init__(self):
        pass

    def forward(self, input):
        """Suorittaa matriisilaskun Wx+b kerroksen läpi"""
        pass

    def backward(self, input, grad_output):
        """Suorittaa gradienttien päivityksen ja palauttaa gradientit seuraavalle kerrokselle"""
        pass


class Tihea(Kerros):  # Vastaa TensorFlow/Keras-kirjastosta kerrosta "Dense"
    """Tiheä kerros RMSProp-optimoijalla"""

    def __init__(self, input_units: NDArray, output_units: NDArray) -> None:
        self.rho = 0.9
        self.epsilon = 1e-7
        self.accumulator_w = np.zeros(shape=(input_units, output_units))
        self.accumulator_b = np.zeros(output_units)
        self.counter = 0
        self.learning_rate = 0.001

        # Painokertoimien alustus
        # https://wandb.ai/sayakpaul/weight-initialization-tb/reports/Effects-of-Weight-Initialization-on-Neural-Networks--Vmlldzo2ODY0NA
        # Edellisen mukaan paras arvo on asettaa ne jakauman mukaan, jossa arvot ovat välillä -y,y ja y = 1/sqrt(input_units)
        _y = 1 / np.sqrt(input_units)
        self.weights = np.random.uniform(
            low=-_y, high=_y, size=(input_units, output_units)
        )
        self.biases = np.zeros(output_units)

    def forward(self, input: NDArray) -> NDArray:
        """Laske kerroksen tulos

        Args:
            input (NDArray): Kerrokselle syötettävä data

        Returns:
            NDArray: Kerroksen tulos Wx + b
        """
        return np.matmul(input, self.weights) + self.biases

    def backward(self, input: NDArray, grad_output: NDArray) -> NDArray:
        # compute d f / d x = d f / d dense * d dense / d x
        # where d dense/ d x = weights transposed
        grad_input = np.dot(grad_output, np.transpose(self.weights))

        # compute gradient w.r.t. weights and biases
        # shape of grad_output is (output_units, input_units)
        grad_weights = np.transpose(np.dot(np.transpose(grad_output), input))
        grad_biases = np.sum(grad_output, axis=0)

        # accumulate gradients
        self.accumulator_w = 0.9 * self.accumulator_w + 0.1 * grad_weights**2
        self.accumulator_b = 0.9 * self.accumulator_b + 0.1 * grad_biases**2

        # update weights and biases
        self.weights = self.weights - (
            self.learning_rate
            * grad_weights
            / (np.sqrt(self.accumulator_w) + self.epsilon)
        )
        self.biases = self.biases - (
            self.learning_rate
            * grad_biases
            / (np.sqrt(self.accumulator_b) + self.epsilon)
        )

        return grad_input


class ReLU(Kerros):
    """ReLU-kerros suorittaa ReLU-aktivointifunktion syötteelle."""

    def __init__(self) -> None:
        pass

    def forward(self, input: NDArray) -> NDArray:
        """Suorita ReLU-funktio alkioittain matriisille [alijoukko, muuttujat]"""
        return np.maximum(0, input)

    def backward(self, input: NDArray, grad_output: NDArray) -> NDArray:
        """Laske gradientti ReLU-kerrokselle

        Args:
            input (NDArray): matriisi muotoa [alijoukko, muuttujat]
            grad_output (NDArray): matriisi muotoa [alijoukko, muuttujat]

        Returns:
            NDArray: gradientit matriisille [alijoukko, muuttujat], johon on laskettu ReLU-funktion gradientti
        """
        relu_grad = input > 0
        return grad_output * relu_grad


def softmax_ristientropia(y_true: NDArray, y_pred: NDArray) -> NDArray:
    """Laske ristientropia softmax-funktioon"""
    logits_for_answers = y_pred[np.arange(len(y_pred)), y_true]
    ristientropia = -logits_for_answers + np.log(np.sum(np.exp(y_pred), axis=-1))
    return ristientropia


def grad_softmax_ristientropia(y_true: NDArray, y_pred: NDArray) -> NDArray:
    """Laske gradientti softmax-ristientropia-funktioon"""
    ones_for_answers = np.zeros_like(y_pred)
    ones_for_answers[np.arange(len(y_pred)), y_true] = 1

    softmax = np.exp(y_pred) / np.exp(y_pred).sum(axis=-1, keepdims=True)

    return (-ones_for_answers + softmax) / y_pred.shape[0]


class Neuroverkko:
    """Neuroverkko RMSProp-optimoijalla

    Attributes:
        network(list): Neuroverkon arkkitehtuuri
    """

    def __init__(self, network: list[Kerros]) -> None:
        """Konstruktori

        Args:
            network (list[Kerros]): Neuroverkon arkkitehtuuri
        """
        self.network = network

    def forward(self, X: NDArray) -> List[NDArray]:
        """
        Laske kaikkien kerrosten aktivoinnit järjestyksessä.
        Palauttaa listan kerroksien aktivoituja dataa.
        """
        activations = []
        for i in range(len(self.network)):
            activations.append(self.network[i].forward(X))
            X = self.network[i].forward(X)

        assert len(activations) == len(self.network)
        return activations

    def predict(self, X: NDArray) -> NDArray:
        """
        Laske neuroverkon päättelemät arvot datalle.

        Args:
            X (NDArray): Data, jonka päättelemät arvot lasketaan.
        """
        logits = self.forward(X)[-1]
        return logits.argmax(axis=-1)

    def train(self, X: NDArray, y: NDArray) -> float:
        """
        Kouluttaa verkkoa alijoukolla X ja vastauksilla y.
        """
        assert len(X) == len(y)

        # Get the layer activations
        layer_activations = self.forward(X)
        y_pred = layer_activations[-1]

        # Compute the loss and the initial gradient
        loss = softmax_ristientropia(y, y_pred)
        loss_grad = grad_softmax_ristientropia(y, y_pred)

        for i in range(1, len(self.network)):
            loss_grad = self.network[len(self.network) - i].backward(
                layer_activations[len(self.network) - i - 1], loss_grad
            )

        return np.mean(loss)

    def fit(
        self,
        X_train: NDArray,
        y_train: NDArray,
        epochs=10,
        batch_size=32,
        validation_data=None,
    ) -> Dict:
        """
        Kouluta neuroverkko.

        Args:
            X_train (NDArray): Data, jonka päättelemät arvot lasketaan.
            y_train (NDArray): X_train:n vastaukset.
            epochs (int): Kuinka monta kertaa koulutus suoritetaan.
            batch_size (int): Kuinka monta dataa koulutetaan kerralla.

            validation_data (tuple): Tuple (X_val, y_val), jossa on validaatiodataa.

        Returns:
            history (dict): Koulutushistoria.
                  Tämä on muotoa:
                  {
                    'loss': [],
                    'accuracy': [],
                    'val_loss': [],
                    'val_accuracy': []
                }
        """

        def iterate_minibatches(
            inputs: NDArray, targets: NDArray, batch_size: int, shuffle=False
        ) -> Iterator[Tuple[NDArray, NDArray]]:
            """Iteroija joka tuottaa alijoukot

            Args:
                inputs (NDArray): data
                targets (NDArray): vastaukset
                batch_size (int): alijoukon koko
                shuffle (bool, optional): Tehdäänkö satunnaisotanta. Defaults to False.

            Yields:
                Iterator[Tuple[NDArray, NDArray]]: Alijoukko (Data, vastaukset)
            """
            assert len(inputs) == len(targets)
            if shuffle:
                indices = np.random.permutation(len(inputs))
            for start_idx in trange(0, len(inputs) - batch_size + 1, batch_size):
                if shuffle:
                    excerpt = indices[start_idx : start_idx + batch_size]
                else:
                    excerpt = slice(start_idx, start_idx + batch_size)
                yield inputs[excerpt], targets[excerpt]

        history = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            loss_values = []
            for x_batch, y_batch in iterate_minibatches(
                X_train, y_train, batch_size=batch_size, shuffle=True
            ):
                loss = self.train(x_batch, y_batch)
                loss_values.append(loss)

            mean_loss = np.mean(loss_values)
            max_loss = np.max(loss_values)
            min_loss = np.min(loss_values)
            history["loss"].append(mean_loss)
            history["accuracy"].append(np.mean(self.predict(X_train) == y_train))
            if validation_data is not None:
                val_acc, val_loss = self.evaluate(
                    validation_data[0], validation_data[1], return_dict=True
                ).values()
                history["val_accuracy"].append(val_acc)
                history["val_loss"].append(val_loss)
            print(
                f"Min loss: {min_loss:.4f}, Max loss: {max_loss:.4f}, Mean loss: {mean_loss:.4f}, Accuracy: {history['accuracy'][-1]:.4f}"
            )
        return history

    def evaluate(self, X, y, return_dict=False):
        """
        Evaluate your network on a dataset.
        """
        if return_dict:
            y_pred = self.forward(X)[-1]
            loss = softmax_ristientropia(y, y_pred)
            return {
                "accuracy": np.mean(y_pred.argmax(axis=-1) == y),
                "loss": np.mean(loss),
            }
        else:
            return np.mean(self.predict(X) == y)
