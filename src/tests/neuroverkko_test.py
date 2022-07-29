import unittest
import neuroverkko as nv
import numpy as np
from tensorflow.keras.losses import SparseCategoricalCrossentropy


class TestCrossEntropy(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([1, 2])
        self.y_pred = np.array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
        self.keras_loss = SparseCategoricalCrossentropy(from_logits=True)
        self.my_loss = nv.softmax_ristientropia

    def test_cross_entropy(self):
        expected_result = self.keras_loss(self.y_true, self.y_pred).numpy()
        result = np.mean(self.my_loss(self.y_true, self.y_pred))
        self.assertAlmostEqual(result, expected_result, places=3)


class TestReLULayer(unittest.TestCase):
    def setUp(self):
        self.layer = nv.ReLU()
        self.x = np.array([[-1, 0, 1, 0.1, 0.5]])
        self.y = np.array([[0, 0, 1, 0.1, 0.5]])
        self.grads = np.array([[-0.5, 0.5, 1, 0.5, 0.5]])

    def test_forward(self):
        np.testing.assert_equal(self.layer.eteenpain(self.x), self.y)

    def test_backward(self):
        np.testing.assert_equal(self.layer.taaksepain(self.x, self.grads), np.array([[0, 0, 1, 0.5, 0.5]]))
