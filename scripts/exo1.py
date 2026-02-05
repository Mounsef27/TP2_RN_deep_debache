# exo1.py - TP2 Exercice 1 : Régression Logistique avec Keras (MNIST)

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

def main():
    # -----------------------
    # 1) Load MNIST
    # -----------------------
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Flatten + normalize
    X_train = X_train.reshape(60000, 784).astype("float32") / 255.0
    X_test  = X_test.reshape(10000, 784).astype("float32") / 255.0

    # One-hot
    Y_train = to_categorical(y_train, 10)
    Y_test  = to_categorical(y_test, 10)

    # -----------------------
    # 2) Build model = Logistic Regression (Softmax Regression)
    # -----------------------
    model = Sequential(name="logistic_regression_mnist")
    model.add(Dense(10, input_dim=784, name="fc1"))  # linear projection
    model.add(Activation("softmax"))                 # softmax

    # Show architecture + nb params
    model.summary()

    # -----------------------
    # 3) Compile
    # -----------------------
    learning_rate = 0.1
    sgd = SGD(learning_rate=learning_rate)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

    # -----------------------
    # 4) Train
    # -----------------------
    batch_size = 100
    nb_epoch = 20
    history = model.fit(
        X_train, Y_train,
        batch_size=batch_size,
        epochs=nb_epoch,
        verbose=1,
        validation_data=(X_test, Y_test)
    )

    # -----------------------
    # 5) Evaluate on test set
    # -----------------------
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("\nTest results:")
    print("%s: %.4f" % (model.metrics_names[0], scores[0]))     # loss
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))  # accuracy

if __name__ == "__main__":
    main()
