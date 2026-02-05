#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
exo2.py — TP2 Deep Learning (Exercice 2)
Perceptron Multi-Couches (MLP) 1 couche cachée avec Keras sur MNIST.

Architecture :
  Dense(100) + sigmoid + Dense(10) + softmax

Entraînement :
  SGD(lr=0.1), categorical_crossentropy, accuracy
  batch_size=100, epochs=20

Sorties :
  figures/exo2_mlp_loss.png
  figures/exo2_mlp_acc.png
  models/mlp_mnist.json
  models/mlp_mnist.weights.h5
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Input
from tensorflow.keras.optimizers import SGD


# -----------------------------
# Utils
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_model_json_h5(model, save_prefix: str) -> None:
    """
    Sauvegarde compatible Keras 3 :
    - architecture : JSON
    - poids : .weights.h5
    """
    json_path = save_prefix + ".json"
    w_path = save_prefix + ".weights.h5"

    with open(json_path, "w", encoding="utf-8") as f:
        f.write(model.to_json())
    model.save_weights(w_path)

    print(f"✅ Model JSON saved to: {json_path}")
    print(f"✅ Weights saved to   : {w_path}")


def plot_history(history, fig_dir: str) -> None:
    """
    Sauvegarde les courbes Loss et Accuracy.
    """
    ensure_dir(fig_dir)

    # LOSS
    plt.figure()
    plt.plot(history.history["loss"], label="train loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(True)
    plt.legend()
    out_loss = os.path.join(fig_dir, "exo2_mlp_loss.png")
    plt.savefig(out_loss, dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Saved:", out_loss)

    # ACC
    plt.figure()
    plt.plot(history.history["accuracy"], label="train acc")
    if "val_accuracy" in history.history:
        plt.plot(history.history["val_accuracy"], label="val acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.grid(True)
    plt.legend()
    out_acc = os.path.join(fig_dir, "exo2_mlp_acc.png")
    plt.savefig(out_acc, dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Saved:", out_acc)


# -----------------------------
# Main
# -----------------------------
def main():
    ROOT = os.getcwd()
    FIG_DIR = os.path.join(ROOT, "figures")
    MODEL_DIR = os.path.join(ROOT, "models")
    ensure_dir(FIG_DIR)
    ensure_dir(MODEL_DIR)

    # 1) Chargement MNIST
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    # 2) Flatten (784) + float32
    X_train = X_train.reshape(X_train.shape[0], 784).astype("float32")
    X_test = X_test.reshape(X_test.shape[0], 784).astype("float32")

    # 3) Normalisation [0,1]
    X_train /= 255.0
    X_test /= 255.0

    # 4) One-hot encoding
    Y_train = tf.keras.utils.to_categorical(y_train, 10)
    Y_test = tf.keras.utils.to_categorical(y_test, 10)

    print("X_train:", X_train.shape, "Y_train:", Y_train.shape)
    print("X_test :", X_test.shape, "Y_test :", Y_test.shape)

    # 5) Modèle MLP (Perceptron) : Dense(100)+sigmoid -> Dense(10)+softmax
    model = Sequential(name="mlp_1hidden_mnist")
    model.add(Input(shape=(784,)))
    model.add(Dense(100, name="fc1"))
    model.add(Activation("sigmoid", name="sigmoid1"))
    model.add(Dense(10, name="fc2"))
    model.add(Activation("softmax", name="softmax"))

    # 6) Summary + vérif params
    model.summary()

    # 7) Compilation
    learning_rate = 0.1
    sgd = SGD(learning_rate=learning_rate)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=sgd,
        metrics=["accuracy"]
    )

    # 8) Entraînement
    batch_size = 100
    nb_epoch = 20

    history = model.fit(
        X_train, Y_train,
        batch_size=batch_size,
        epochs=nb_epoch,
        verbose=1,
        validation_split=0.2
    )

    # 9) Évaluation
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print(f"\nTest loss: {scores[0]:.4f}")
    print(f"Test accuracy: {scores[1]*100:.2f}%")

    # 10) Plots
    plot_history(history, FIG_DIR)

    # 11) Sauvegarde modèle
    save_prefix = os.path.join(MODEL_DIR, "mlp_mnist")
    save_model_json_h5(model, save_prefix)


if __name__ == "__main__":
    # Réduire les logs TF
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()
