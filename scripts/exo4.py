#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
exo4.py — TP2 Deep Learning (Exercice 4)
Visualisation MNIST test en 2D via t-SNE et PCA + métriques de séparabilité :
- Convex Hulls
- Ellipses (GaussianMixture)
- Neighborhood Hit (NH)

Sorties :
  figures/exo4_tsne.png
  figures/exo4_pca.png
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
from sklearn.decomposition import PCA

import tensorflow as tf


# -----------------------------
# 1) Métriques demandées
# -----------------------------
def convexHulls(points, labels, n_classes=10):
    """Convex hull par classe."""
    convex_hulls = []
    for i in range(n_classes):
        pts = points[labels == i, :]
        convex_hulls.append(ConvexHull(pts))
    return convex_hulls


def best_ellipses(points, labels, n_classes=10):
    """GMM 1 composante par classe => ellipse (covariance full)."""
    gaussians = []
    for i in range(n_classes):
        pts = points[labels == i, :]
        gm = GaussianMixture(n_components=1, covariance_type="full", random_state=0).fit(pts)
        gaussians.append(gm)
    return gaussians


def neighboring_hit(points, labels, k=6, n_classes=10):
    """
    Neighborhood Hit (NH) :
    - NH global = moyenne sur tous les points
    - NH par classe = moyenne sur chaque classe
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(points)
    distances, indices = nbrs.kneighbors(points)

    txs = 0.0
    txsc = [0.0] * n_classes
    nppts = [0.0] * n_classes

    for i in range(len(points)):
        tx = 0.0
        for j in range(1, k + 1):  # ignore j=0 (le point lui-même)
            if labels[indices[i, j]] == labels[i]:
                tx += 1
        tx /= k

        c = int(labels[i])
        txsc[c] += tx
        nppts[c] += 1
        txs += tx

    for c in range(n_classes):
        if nppts[c] > 0:
            txsc[c] /= nppts[c]

    return txs / len(points), txsc


# -----------------------------
# 2) Visualisation (3 panneaux)
# -----------------------------
def visualization(points2D, labels, convex_hulls, ellipses, projname, nh, out_png):
    """
    3 panneaux:
      (1) scatter
      (2) convex hulls
      (3) ellipses GMM
    """
    cmap = cm.tab10
    vals = [i / 10.0 for i in range(10)]
    points2D_c = [points2D[labels == i, :] for i in range(10)]

    plt.figure(figsize=(6, 12), dpi=120)
    plt.subplots_adjust(hspace=0.35)

    # (1) Scatter
    ax1 = plt.subplot(311)
    sc = ax1.scatter(points2D[:, 0], points2D[:, 1],
                     c=labels, s=4, edgecolors="none",
                     cmap=cmap, alpha=0.9)
    plt.colorbar(sc, ax=ax1, ticks=range(10))
    ax1.set_title(f"2D {projname} - NH = {nh*100:.2f}%")
    ax1.set_xticks([]); ax1.set_yticks([])

    # (2) Convex Hulls
    ax2 = plt.subplot(312)
    for i in range(10):
        ch = np.append(convex_hulls[i].vertices, convex_hulls[i].vertices[0])
        ax2.plot(points2D_c[i][ch, 0], points2D_c[i][ch, 1],
                 "-", color=cmap(vals[i]), linewidth=2)
    ax2.set_title(f"{projname} - Convex Hulls")
    ax2.set_xticks([]); ax2.set_yticks([])

    # (3) Ellipses (GMM)
    ax3 = plt.subplot(313)
    for i in range(10):
        X = points2D[labels == i, :]
        gm = ellipses[i]
        mean = gm.means_[0]
        cov = gm.covariances_[0]

        # Diagonalisation covariance => ellipse
        v, w = linalg.eigh(cov)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0]) * 180.0 / np.pi

        ax3.scatter(X[:, 0], X[:, 1], s=3, alpha=0.15, color=cmap(vals[i]))
        ell = mpl.patches.Ellipse(
            xy=mean, width=v[0], height=v[1],
            angle=180.0 + angle,              # IMPORTANT: angle= nommé
            color=cmap(vals[i]), alpha=0.35
        )
        ax3.add_patch(ell)

    ax3.set_title(f"{projname} - Best fitting ellipses (GMM)")
    ax3.set_xticks([]); ax3.set_yticks([])

    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print("✅ Saved:", out_png)


# -----------------------------
# 3) Pipeline principal exo4
# -----------------------------
def main():
    ROOT = os.getcwd()
    FIG_DIR = os.path.join(ROOT, "figures")
    os.makedirs(FIG_DIR, exist_ok=True)

    # ---- Chargement MNIST test
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # ---- Préparation : flatten + normalisation
    X_test = x_test.reshape((x_test.shape[0], 784)).astype("float32") / 255.0
    labels = y_test.astype(int)

    print("MNIST test:", X_test.shape, labels.shape)

    # ---- PCA 2D
    print("\n=== PCA 2D ===")
    pca = PCA(n_components=2, random_state=42)
    x2d_pca = pca.fit_transform(X_test)
    nh_pca, nh_pca_by_class = neighboring_hit(x2d_pca, labels, k=6)
    print("NH global PCA:", nh_pca)
    print("NH per class PCA:", np.round(nh_pca_by_class, 3))

    conv_pca = convexHulls(x2d_pca, labels)
    ell_pca = best_ellipses(x2d_pca, labels)
    out_pca = os.path.join(FIG_DIR, "exo4_pca.png")
    visualization(x2d_pca, labels, conv_pca, ell_pca, "PCA", nh_pca, out_pca)

    # ---- t-SNE 2D
    print("\n=== t-SNE 2D ===")
    t0 = time.time()
    tsne = TSNE(n_components=2, init="pca", perplexity=30, verbose=2, random_state=42)
    x2d_tsne = tsne.fit_transform(X_test)
    dt = time.time() - t0
    print(f"t-SNE time: {dt:.2f} s")

    nh_tsne, nh_tsne_by_class = neighboring_hit(x2d_tsne, labels, k=6)
    print("NH global t-SNE:", nh_tsne)
    print("NH per class t-SNE:", np.round(nh_tsne_by_class, 3))

    conv_tsne = convexHulls(x2d_tsne, labels)
    ell_tsne = best_ellipses(x2d_tsne, labels)
    out_tsne = os.path.join(FIG_DIR, "exo4_tsne.png")
    visualization(x2d_tsne, labels, conv_tsne, ell_tsne, "t-SNE", nh_tsne, out_tsne)

    print("\n✅ Exo4 terminé.")
    print("Figures générées dans:", FIG_DIR)


if __name__ == "__main__":
    # (optionnel) réduire les logs TF
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()
