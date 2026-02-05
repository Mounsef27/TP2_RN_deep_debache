# TP2 — Deep Learning avec Keras & Manifold Untangling (US3362 CNAM)

Ce dépôt contient mon travail pour le **TP2** du cours **US3362 (CNAM)** : prise en main de **Keras/TensorFlow** sur **MNIST**, entraînement de modèles (régression logistique, MLP, CNN type LeNet) et **visualisation** des représentations (t-SNE / PCA) avec mesures de séparabilité (Convex Hulls, ellipses Gaussiennes, Neighborhood Hit).

---

## 📌 Objectifs du TP

- Implémenter et entraîner avec **Keras** :
  - **Exo1** : Régression logistique (Dense(10) + softmax)
  - **Exo2** : MLP (Dense(100) + sigmoid + Dense(10) + softmax)
  - **Exo3** : CNN type **LeNet-like**
- Visualiser la séparabilité des classes avec :
  - **Exo4** : t-SNE vs PCA sur les données brutes (MNIST test)
  - **Exo5** : t-SNE sur les **représentations latentes** (MLP hidden vs CNN latent)
- Mesurer la qualité de séparation via :
  - **Convex Hulls**
  - **Ellipses (GaussianMixture)**
  - **Neighborhood Hit (NH)**

---

## 📂 Structure du projet

```text
TP2_US3362_deep_keras/
├─ exo2.py                     # MLP Keras + entraînement + courbes + sauvegarde
├─ exo3.py                     # CNN LeNet-like + entraînement + timing + sauvegarde
├─ exo4.py                     # t-SNE vs PCA + métriques + figures
├─ exo5.py                     # t-SNE espaces latents (MLP vs CNN) + métriques
├─ scripts/
│  └─ make_all.sh              # Lance tous les scripts et génère figures/models/logs
├─ notebooks/
│  └─ TP2.ipynb                # Notebook de travail
├─ figures/                    # Figures générées (loss/acc, t-SNE, PCA, etc.)
├─ models/                     # Modèles sauvegardés
│  ├─ mlp_mnist.json
│  ├─ mlp_mnist.weights.h5
│  └─ exo3_lenet_like_mnist.keras
├─ logs/                       # Logs d'exécution (optionnel)
└─ requirements.txt


## ⚙️ Installation (environnement isolé)

### 1) Créer un environnement Python (recommandé)
```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
