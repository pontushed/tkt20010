import unittest
import neuroverkko as nv
import numpy as np
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import os


class TestCrossEntropy(unittest.TestCase):
    """Testataan ristientropian laskeminen käyttäen referenssinä kerasin funktiota"""

    def setUp(self):
        self.y_true = np.array([1, 2])
        self.y_pred = np.array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
        self.keras_loss = SparseCategoricalCrossentropy(from_logits=True)
        self.my_loss = nv.softmax_ristientropia
        self.my_loss_grad = nv.grad_softmax_ristientropia

    def test_ristientropia(self):
        expected_result = self.keras_loss(self.y_true, self.y_pred).numpy()
        result = np.mean(self.my_loss(self.y_true, self.y_pred))
        self.assertAlmostEqual(result, expected_result, places=3)

    def test_grad_softmax_ristientropia(self):
        expected_result = np.array([[0.1133573, -0.2211861, 0.1078288], [0.1245717, 0.2508566, -0.3754283]])
        result = self.my_loss_grad(self.y_true, self.y_pred)
        np.testing.assert_almost_equal(result, expected_result)


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

    def test_description(self):
        self.assertEqual(str(self.layer), "ReLU-kerros")


class TestSoftmax(unittest.TestCase):
    """Softmax-funktio"""

    def test_softmax(self):
        x = np.array([[1, 3, 2]])
        y = np.array([[0.09003057, 0.66524096, 0.24472847]])
        np.testing.assert_array_almost_equal(nv.softmax(x), y)
        np.testing.assert_almost_equal(np.sum(nv.softmax(x)), 1)


class TestKerros(unittest.TestCase):
    """Testataan abstrakti kerros-luokka"""

    def setUp(self):
        self.layer = nv.Kerros()

    def test_kerros(self):
        self.assertIsInstance(self.layer, nv.Kerros)
        self.assertTrue("eteenpain" in dir(self.layer))
        self.assertTrue("taaksepain" in dir(self.layer))
        self.layer.eteenpain(None)
        self.layer.taaksepain(None, None)


class TestTihea(unittest.TestCase):
    """Testataan tiheä-kerros"""

    def setUp(self):
        self.layer = nv.Tihea(2, 2)

    def test_tihea(self):
        self.assertIsInstance(self.layer, nv.Tihea)
        self.assertTrue("eteenpain" in dir(self.layer))
        self.assertTrue("taaksepain" in dir(self.layer))
        self.assertEqual(str(self.layer), "Tiheä kerros (2)")

    def test_eteenpain(self):
        x = np.array([[1, 2]])
        painot = self.layer.painot
        vakiot = self.layer.vakiot
        expected = x @ painot + vakiot
        np.testing.assert_equal(self.layer.eteenpain(x), expected)

    def test_taaksepain(self):
        x = np.array([[1, 2]])
        grads = np.array([[1, 1]])
        painot = self.layer.painot
        # Funktion tulisi palauttaa gradienttien ja painojen pistetulo
        expected = np.dot(grads, np.transpose(painot))
        np.testing.assert_equal(self.layer.taaksepain(x, grads), expected)
        # Painojen tulisi muuttua
        np.testing.assert_equal(np.any(np.not_equal(self.layer.painot, painot)), True)


class TestNeuroverkko(unittest.TestCase):
    def setUp(self):
        self.verkko = nv.Neuroverkko([nv.Tihea(2, 2), nv.ReLU()])
        self.X = np.array([[1, 2], [3, 4]])

    def test_verkko(self):
        self.assertIsInstance(self.verkko, nv.Neuroverkko)
        self.assertEqual(str(self.verkko), "Tiheä kerros (2) -> ReLU-kerros")
        self.assertTrue("lataa" in dir(self.verkko))
        self.assertTrue("tallenna" in dir(self.verkko))
        self.assertTrue("eteenpain" in dir(self.verkko))
        self.assertTrue("ennusta" in dir(self.verkko))
        self.assertTrue("kouluta" in dir(self.verkko))
        self.assertTrue("sovita" in dir(self.verkko))
        self.assertTrue("evaluoi" in dir(self.verkko))

    def test_tallenna_ja_lataa(self):
        self.verkko.verkko[0].painot = np.array([[1, 0], [0, 1]])
        self.verkko.verkko[0].vakiot = np.array([3, 4])
        self.verkko.tallenna("testi.pkl")
        self.verkko.verkko[0].painot = np.array([[0, 1], [1, 0]])
        self.verkko.verkko[0].vakiot = np.array([0, 0])
        self.verkko.lataa("testi.pkl")
        np.testing.assert_equal(self.verkko.verkko[0].painot, np.array([[1, 0], [0, 1]]))
        np.testing.assert_equal(self.verkko.verkko[0].vakiot, np.array([3, 4]))
        os.remove("testi.pkl")

    def test_eteenpain(self):
        self.verkko.verkko[0].painot = np.array([[1, 0], [0, 1]])
        prediction = self.verkko.eteenpain(self.X)
        np.testing.assert_equal(prediction, np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]]))

    def test_ennusta_todennakoisyydet(self):
        self.verkko.verkko[0].painot = np.array([[1, 0], [0, 1]])
        prediction = self.verkko.ennusta(self.X, todennakoisyydet=True)
        np.testing.assert_almost_equal(prediction, np.array([[0.27, 0.73], [0.27, 0.73]]), decimal=2)

    def test_ennusta(self):
        self.verkko.verkko[0].painot = np.array([[1, 0], [0, 1]])
        prediction = self.verkko.ennusta(self.X)
        np.testing.assert_equal(prediction, np.array([1, 1]))

    def test_evaluoi(self):
        self.verkko.verkko[0].painot = np.array([[1, 0], [0, 1]])
        self.verkko.verkko[0].vakiot = np.array([3, 4])
        self.verkko.verkko[1].painot = np.array([[1, 0], [0, 1]])
        self.verkko.verkko[1].vakiot = np.array([3, 4])
        predictions = self.verkko.evaluoi(self.X, np.array([1, 1]))
        np.testing.assert_equal(predictions, 1.0)

    def test_evaluoi_dict(self):
        self.verkko.verkko[0].painot = np.array([[1, 0], [0, 1]])
        self.verkko.verkko[0].vakiot = np.array([3, 4])
        self.verkko.verkko[1].painot = np.array([[1, 0], [0, 1]])
        self.verkko.verkko[1].vakiot = np.array([3, 4])
        predictions = self.verkko.evaluoi(self.X, np.array([1, 1]), return_dict=True)
        self.assertAlmostEqual(predictions["hukka"], 0.13, places=2)
        self.assertEqual(predictions["tarkkuus"], 1.0)

    def test_kouluta(self):
        self.verkko.verkko[0].painot = np.array([[1, 0], [0, 1]])
        self.verkko.verkko[0].vakiot = np.array([3, 4])
        self.verkko.verkko[1].painot = np.array([[1, 0], [0, 1]])
        self.verkko.verkko[1].vakiot = np.array([3, 4])
        hukka = self.verkko.kouluta(self.X, np.array([1, 1]))
        self.assertAlmostEqual(hukka, 0.13, places=2)

    def test_sovita(self):
        X = np.arange(0, 20, 1).reshape((10, 2))
        y = np.ones(shape=(10, 1), dtype=np.int8)
        self.verkko.verkko[0].painot = np.array([[1, 0], [0, 1]])
        self.verkko.verkko[0].vakiot = np.array([0, 0])
        historia = self.verkko.sovita(X, y, epookit=1, alijoukon_koko=2)
        self.assertLess(historia["hukka"][-1], 0.3)
        self.assertAlmostEqual(historia["tarkkuus"][-1], 1.0, places=2)

    def test_sovita_validaatiodatalla(self):
        X = np.arange(0, 20, 1).reshape((10, 2))
        X_val = np.arange(0, 20, 2).reshape((5, 2))
        y = np.ones(shape=(10, 1), dtype=np.int8)
        y_val = np.ones(shape=(5, 1), dtype=np.int8)
        self.verkko.verkko[0].painot = np.array([[1, 0], [0, 1]])
        self.verkko.verkko[0].vakiot = np.array([0, 0])
        historia = self.verkko.sovita(X, y, epookit=1, alijoukon_koko=2, validaatiodata=(X_val, y_val))
        self.assertEqual(historia["tarkkuus"][-1], 1.0)
        self.assertEqual(historia["validointitarkkuus"][-1], 1.0)
        self.assertIsNotNone(historia["validointihukka"])
        self.assertIsNotNone(historia["validointitarkkuus"])

    def test_sovita_ala_sekoita(self):
        X = np.arange(0, 20, 1).reshape((10, 2))
        y = np.ones(shape=(10, 1), dtype=np.int8)
        self.verkko.verkko[0].painot = np.array([[1, 0], [0, 1]])
        self.verkko.verkko[0].vakiot = np.array([0, 0])
        historia = self.verkko.sovita(X, y, epookit=1, alijoukon_koko=2, sekoita=False)
        self.assertLess(historia["hukka"][-1], 0.3)
        self.assertEqual(historia["tarkkuus"][-1], 1.0)
