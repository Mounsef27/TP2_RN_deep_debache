#!/usr/bin/env python3
# exo3.py - TP2 Exercice 3 : ConvNet (LeNet-like) avec Keras sur MNIST

import os, time
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import Callback

class EpochTimer(Callback):
    def on_train_begin(self, logs=None):
        self.epoch_times = []
    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.time()
    def on_epoch_end(self, epoch, logs=None):
        self.epoch_times.append(time.time() - self.t0)

def saveModelJSON(model, savename):
    with open(savename + ".json", "w") as f:
        f.write(model.to_json())
    model.save_weights(savename + ".weights.h5")

def main():
    os.makedirs("figures", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Load MNIST
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape (N, 28, 28, 1) + normalize
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype("float32") / 255.0
    x_test  = x_test.reshape(x_test.shape[0], 28, 28, 1).astype("float32") / 255.0
    input_shape = (28, 28, 1)

    # One-hot
    Y_train = to_categorical(y_train, 10)
    Y_test  = to_categorical(y_test, 10)

    # Build LeNet-like model
    model = Sequential(name="lenet_like_mnist")
    model.add(Conv2D(16, kernel_size=(5,5), padding="valid", input_shape=input_shape, name="conv1"))
    model.add(Activation("relu", name="relu1"))
    model.add(MaxPooling2D(pool_size=(2,2), name="pool1"))

    model.add(Conv2D(32, kernel_size=(5,5), padding="valid", name="conv2"))
    model.add(Activation("relu", name="relu2"))
    model.add(MaxPooling2D(pool_size=(2,2), name="pool2"))

    model.add(Flatten(name="flatten"))
    model.add(Dense(100, name="fc1"))
    model.add(Activation("sigmoid", name="sigmoid1"))
    model.add(Dense(10, name="fc2"))
    model.add(Activation("softmax", name="softmax"))

    model.summary()

    # Compile
    lr = 0.1
    model.compile(
        loss="categorical_crossentropy",
        optimizer=SGD(learning_rate=lr),
        metrics=["accuracy"]
    )

    # Train with timing
    timer = EpochTimer()
    batch_size = 100
    nb_epoch = 20

    hist = model.fit(
        x_train, Y_train,
        batch_size=batch_size,
        epochs=nb_epoch,
        verbose=1,
        validation_data=(x_test, Y_test),
        callbacks=[timer]
    )

    avg_epoch = float(np.mean(timer.epoch_times))
    print(f"\nAverage epoch time: {avg_epoch:.3f} s/epoch")

    # Evaluate
    scores = model.evaluate(x_test, Y_test, verbose=0)
    print("Test loss: %.4f" % scores[0])
    print("Test accuracy: %.2f%%" % (scores[1]*100))

    # Figures
    plt.figure()
    plt.plot(hist.history["loss"], label="train loss")
    plt.plot(hist.history["val_loss"], label="val loss")
    plt.xlabel("epoch"); plt.ylabel("loss")
    plt.grid(True); plt.legend()
    plt.savefig("figures/exo3_cnn_loss.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(hist.history["accuracy"], label="train acc")
    plt.plot(hist.history["val_accuracy"], label="val acc")
    plt.xlabel("epoch"); plt.ylabel("accuracy")
    plt.grid(True); plt.legend()
    plt.savefig("figures/exo3_cnn_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Save model (modern .keras)
    model.save("models/exo3_lenet_like_mnist.keras")
    print("Saved model: models/exo3_lenet_like_mnist.keras")

    # Optional JSON+H5
    saveModelJSON(model, "models/exo3_lenet_like_mnist")
    print("Saved model JSON+weights: models/exo3_lenet_like_mnist.json + .weights.h5")

if __name__ == "__main__":
    main()
