# TP 2 : Optimisation et Cycle de Vie des Modèles de Deep Learning

[cite_start]Ce projet se concentre sur l'amélioration des performances d'un classifieur de chiffres manuscrits (MNIST) en appliquant des techniques avancées d'ingénierie du Deep Learning[cite: 5]. L'objectif est de diagnostiquer les problèmes d'apprentissage et de mettre en œuvre des solutions pour garantir une généralisation optimale sur des données inconnues.

## 🎯 Objectifs Pédagogiques
- Diagnostiquer la performance d'un modèle (Biais vs Variance).
- Maîtriser les techniques de régularisation (L2 et Dropout).
- Accélérer la convergence avec la Batch Normalization.
- Industrialiser le suivi des expérimentations avec MLflow.

## 🏗️ Architecture du Modèle
Le modèle implémenté est un réseau de neurones multicouches (MLP) intégrant :
- **Entrée** : Vecteur de 784 pixels normalisés.
- **Double Dropout** : Une couche après l'entrée et une avant la sortie (taux 0.2) pour prévenir le surapprentissage.
- **Régularisation L2** : Pénalité de 0.001 sur les poids de la couche Dense.
- **Batch Normalization** : Stabilisation des activations avant la fonction ReLU.
- **Sortie** : Couche Softmax pour une classification sur 10 classes.

## 🧪 Expérimentations et Résultats
Nous avons utilisé MLflow pour comparer l'impact des différents optimiseurs sur 5 époques.



### Comparaison des Optimiseurs
| Optimiseur | Accuracy (Val) | Loss (Val) | Observation |
|------------|----------------|------------|-------------|
| **SGD with Momentum** | **97.90%** | 0.3557 | Meilleure généralisation globale. |
| **Adam** | 97.42% | 0.1830 | Convergence très rapide. |
| **RMSprop** | 97.45% | 0.1714 | Perte la plus faible en validation. |

### Diagnostic Biais/Variance
L'écart réduit entre la précision d'entraînement (95.94%) et de validation (97.90%) démontre que les techniques de régularisation ont efficacement éliminé le risque de surapprentissage (High Variance).

## 🛠️ Installation et Exécution
1. Cloner le dépôt.
2. Installer les dépendances :
   ```bash
   pip install tensorflow numpy mlflow
3. Exécuter le script d'entraînement :
   ```bash
   python train_model_v2.py
4. Lancer l'interface de suivi MLflow :
   ```bash
   mlflow ui
