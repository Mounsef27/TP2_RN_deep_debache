#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
exo5.py — TP2 Deep Learning (Exercice 5)
Visualisation des représentations internes (MLP vs CNN) avec t-SNE + métriques:
- Convex Hulls
- Ellipses (GaussianMixture)
- Neighborhood Hit (NH)

Fichiers attendus :
  models/mlp_mnist.json
  models/mlp_mnist.weights.h5
  models/exo3_lenet_like_mnist.keras

Sorties :
  figures/exo5_tsne_mlp_latent.png
  figures/exo5_tsne_cnn_latent.png
"""

import os
import time
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.spatial import ConvexHull
from scipy import linalg
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE

import tensorflow as tf


# -----------------------------
# Utils: métriques et affichage
# -----------------------------
def convexHulls(points, labels, n_classes=10):
    convex_hulls = []
    for i in range(n_classes):
        pts = points[labels == i, :]
        convex_hulls.append(ConvexHull(pts))
    return convex_hulls


def best_ellipses(points, labels, n_classes=10):
    gaussians = []
    for i in range(n_classes):
        pts = points[labels == i, :]
        gm = GaussianMixture(n_components=1, covariance_type="full", random_state=0).fit(pts)
        gaussians.append(gm)
    return gaussians


def neighboring_hit(points, labels, k=6):
    """
    NH global + NH par classe.
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(points)
    distances, indices = nbrs.kneighbors(points)

    txs = 0.0
    txsc = [0.0] * 10
    nppts = [0.0] * 10

    for i in range(len(points)):
        tx = 0.0
        for j in range(1, k + 1):  # ignore j=0 (le point lui-même)
            if labels[indices[i, j]] == labels[i]:
                tx += 1
        tx /= k

        txsc[int(labels[i])] += tx
        nppts[int(labels[i])] += 1
        txs += tx

    for c in range(10):
        if nppts[c] > 0:
            txsc[c] /= nppts[c]

    return txs / len(points), txsc


def visualization(points2D, labels, convex_hulls, ellipses, projname, nh, out_png):
    """
    3 panneaux: scatter + convex hulls + ellipses (GMM)
    Compatible Matplotlib récent : Ellipse(angle=...) en argument nommé.
    """
    cmap = cm.tab10
    vals = [i / 10.0 for i in range(10)]

    points2D_c = [points2D[labels == i, :] for i in range(10)]

    plt.figure(figsize=(6, 12), dpi=120)
    plt.subplots_adjust(hspace=0.35)

    # 1) Scatter
    ax1 = plt.subplot(311)
    sc = ax1.scatter(points2D[:, 0], points2D[:, 1],
                     c=labels, s=4, edgecolors="none",
                     cmap=cmap, alpha=0.9)
    plt.colorbar(sc, ax=ax1, ticks=range(10))
    ax1.set_title(f"2D {projname} - NH = {nh*100:.2f}%")
    ax1.set_xticks([]); ax1.set_yticks([])

    # 2) Convex hulls
    ax2 = plt.subplot(312)
    for i in range(10):
        ch = np.append(convex_hulls[i].vertices, convex_hulls[i].vertices[0])
        ax2.plot(points2D_c[i][ch, 0], points2D_c[i][ch, 1],
                 "-", color=cmap(vals[i]), linewidth=2)
    ax2.set_title(f"{projname} - Convex Hulls")
    ax2.set_xticks([]); ax2.set_yticks([])

    # 3) Ellipses (GMM)
    ax3 = plt.subplot(313)
    for i in range(10):
        X = points2D[labels == i, :]
        gm = ellipses[i]
        mean = gm.means_[0]
        cov = gm.covariances_[0]

        v, w = linalg.eigh(cov)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])

        angle = np.arctan2(u[1], u[0]) * 180.0 / np.pi

        ax3.scatter(X[:, 0], X[:, 1], s=3, alpha=0.15, color=cmap(vals[i]))
        ell = mpl.patches.Ellipse(
            xy=mean, width=v[0], height=v[1], angle=180.0 + angle,
            color=cmap(vals[i]), alpha=0.35
        )
        ax3.add_patch(ell)

    ax3.set_title(f"{projname} - Best fitting ellipses (GMM)")
    ax3.set_xticks([]); ax3.set_yticks([])

    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Saved:", out_png)


# -----------------------------
# Chargement MNIST + préparation
# -----------------------------
def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalisation [0,1]
    x_test = x_test.astype("float32") / 255.0

    # Version CNN: (N,28,28,1)
    x_test_cnn = np.expand_dims(x_test, axis=-1)

    # Version MLP: (N,784)
    x_test_fc = x_test.reshape((x_test.shape[0], 784))

    return x_test_fc, x_test_cnn, y_test


# -----------------------------
# Chargement des modèles
# -----------------------------
def load_mlp(json_path, weights_path):
    with open(json_path, "r", encoding="utf-8") as f:
        mlp = tf.keras.models.model_from_json(f.read())
    mlp.load_weights(weights_path)
    return mlp


def load_cnn(cnn_path):
    return tf.keras.models.load_model(cnn_path)


# -----------------------------
# Extraction latente
# -----------------------------
def build_embedder_mlp(mlp):
    """
    On veut la représentation cachée (Dense 100).
    On repère la première Dense avec units=100.
    """
    # Important: pour que mlp.input existe, on "build" le modèle
    mlp.build((None, 784))

    hidden_name = None
    for layer in mlp.layers:
        if isinstance(layer, tf.keras.layers.Dense) and getattr(layer, "units", None) == 100:
            hidden_name = layer.name
            break
    if hidden_name is None:
        raise ValueError("Impossible de trouver une Dense(units=100) dans le MLP.")

    embedder = tf.keras.Model(inputs=mlp.input,
                              outputs=mlp.get_layer(hidden_name).output)
    return embedder, hidden_name


def build_embedder_cnn(cnn):
    """
    Pour le CNN, on prend une représentation "latente" robuste :
    - idéalement la couche 'flatten'
    - sinon une Dense(100) si flatten n'existe pas
    """
    # build si nécessaire
    cnn.build((None, 28, 28, 1))

    target_name = None

    # 1) privilégie Flatten
    for layer in cnn.layers:
        if isinstance(layer, tf.keras.layers.Flatten):
            target_name = layer.name
            break

    # 2) sinon Dense(100)
    if target_name is None:
        for layer in cnn.layers:
            if isinstance(layer, tf.keras.layers.Dense) and getattr(layer, "units", None) == 100:
                target_name = layer.name
                break

    if target_name is None:
        raise ValueError("Impossible de trouver Flatten ou Dense(100) dans le CNN.")

    embedder = tf.keras.Model(inputs=cnn.input,
                              outputs=cnn.get_layer(target_name).output)
    return embedder, target_name


# -----------------------------
# Pipeline complet (MLP ou CNN)
# -----------------------------
def run_tsne_and_metrics(H, labels, fig_path, title):
    print(f"\n=== t-SNE: {title} ===")
    t0 = time.time()
    tsne = TSNE(n_components=2, init="pca", perplexity=30, verbose=2, random_state=42)
    H2d = tsne.fit_transform(H)
    dt = time.time() - t0
    print(f"t-SNE time: {dt:.2f} s  | H2d shape: {H2d.shape}")

    # métriques
    conv = convexHulls(H2d, labels)
    ell = best_ellipses(H2d, labels)
    nh_global, nh_by_class = neighboring_hit(H2d, labels, k=6)

    print(f"NH global {title}: {nh_global:.6f} ({nh_global*100:.2f}%)")
    print("NH per class:", np.round(nh_by_class, 3))

    # figure
    visualization(H2d, labels, conv, ell, title, nh_global, fig_path)

    return H2d, nh_global, nh_by_class


def main():
    # Racine projet = dossier courant
    ROOT = os.getcwd()

    MODELS_DIR = os.path.join(ROOT, "models")
    FIG_DIR = os.path.join(ROOT, "figures")
    os.makedirs(FIG_DIR, exist_ok=True)

    # Paths modèles (selon ton projet)
    MLP_JSON = os.path.join(MODELS_DIR, "mlp_mnist.json")
    MLP_W = os.path.join(MODELS_DIR, "mlp_mnist.weights.h5")
    CNN_PATH = os.path.join(MODELS_DIR, "exo3_lenet_like_mnist.keras")

    # Vérif fichiers
    for p in [MLP_JSON, MLP_W, CNN_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Fichier introuvable: {p}")

    # Données
    X_test_fc, X_test_cnn, y_test = load_mnist()
    labels = y_test.astype(int)

    # ---------------- MLP ----------------
    print("\n====================")
    print("Chargement MLP ...")
    mlp = load_mlp(MLP_JSON, MLP_W)
    print("MLP loaded ✅")
    mlp.summary()

    mlp_embedder, mlp_layer = build_embedder_mlp(mlp)
    print(f"MLP latent layer: {mlp_layer}")
    H_mlp = mlp_embedder.predict(X_test_fc, batch_size=256, verbose=0)
    print("H_mlp:", H_mlp.shape)

    out_mlp = os.path.join(FIG_DIR, "exo5_tsne_mlp_latent.png")
    run_tsne_and_metrics(H_mlp, labels, out_mlp, "t-SNE MLP hidden (H)")

    # ---------------- CNN ----------------
    print("\n====================")
    print("Chargement CNN ...")
    cnn = load_cnn(CNN_PATH)
    print("CNN loaded ✅")
    cnn.summary()

    cnn_embedder, cnn_layer = build_embedder_cnn(cnn)
    print(f"CNN latent layer: {cnn_layer}")
    H_cnn = cnn_embedder.predict(X_test_cnn, batch_size=256, verbose=0)
    print("H_cnn:", H_cnn.shape)

    out_cnn = os.path.join(FIG_DIR, "exo5_tsne_cnn_latent.png")
    run_tsne_and_metrics(H_cnn, labels, out_cnn, "t-SNE CNN latent")

    print("\n✅ Exo5 terminé. Figures dans:", FIG_DIR)


if __name__ == "__main__":
    # Optionnel: réduire un peu la verbosité TF
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()
