from model import train_model
from sklearn.datasets import load_iris

def test_model_training():
    model, accuracy = train_model()

    assert model is not None
    assert 0 <= accuracy <= 1

def test_iris_dataset():
    iris = load_iris()

    assert iris.data.shape == (150, 4)
    assert len(iris.target) == 150

def test_model_prediction():
    model, accuracy = train_model()

    prediction = model.predict([[5.1, 3.5, 1.4, 0.2]])

    assert prediction[0] in [0, 1, 2]
