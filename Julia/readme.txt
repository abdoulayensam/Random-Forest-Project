
# README

## Projet : Implémentation et Visualisation de Forêts Aléatoires en Julia

### Structure du Projet
Le projet est organisé de manière simple et directe :

1. **Code Source :**
   - **`Source_code.jl`** : Fichier unique contenant l'intégralité du code. Ce fichier gère la génération des données, l'implémentation des forêts aléatoires, ainsi que l'affichage des résultats.

2. **Données :**
   - **`apprenants_dataset.csv`** : Jeu de données généré ou importé pour tester le modèle.

3. **Visualisations :**
   - Les graphiques générés sont automatiquement sauvegardés dans le dossier `plots/`.

---

### Objectifs du Projet

1. **Modélisation avec Forêts Aléatoires :**
   - Implémentation d'un modèle de forêt aléatoire.
   - Construction et entraînement d'arbres de décision sur des échantillons aléatoires.
   - Agrégation des prédictions via le vote majoritaire.

2. **Analyse des Données :**
   - Génération et visualisation d'un dataset synthétique représentatif.
   - Exploration des caractéristiques des données à l'aide de statistiques descriptives et de graphiques.

3. **Visualisation des Résultats :**
   - Graphiques des arbres de décision.
   - Histogrammes des caractéristiques.
   - Matrice de corrélation et importance des caractéristiques.

---

### Instructions d'Exécution

#### Prérequis
- Installer **Julia** (version >= 1.8).
- Installer les bibliothèques nécessaires en exécutant :
  ```julia
  using Pkg
  Pkg.add(["DataFrames", "CSV", "StatsBase", "Plots", "StatsPlots", "PrettyTables", "GraphRecipes", "Random"])
  ```

#### Étapes

1. **Télécharger ou cloner le projet :**
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Exécuter le projet :**
   Lancer le fichier principal **`main.jl`** dans Julia :
   ```bash
   julia main.jl
   ```

3. **Résultats :**
   - Les résultats statistiques et les prédictions seront affichés dans la console.
   - Les visualisations des arbres et des données seront sauvegardées dans le dossier `plots/`.

---

### Fonctionnalités Implémentées

1. **Forêt Aléatoire :**
   - Construction de plusieurs arbres sur des sous-ensembles aléatoires.
   - Agrégation des prédictions via le vote majoritaire.
   - Support des données catégoriques et numériques.

2. **Visualisation des Données et Modèles :**
   - Tracé des arbres de décision.
   - Distribution des types d'apprenants.
   - Importance des caractéristiques.

---

### Notes
- Le fichier de données **`apprenants_dataset.csv`** sera généré automatiquement dans le répertoire.
- En cas de problème, assurez-vous que toutes les bibliothèques sont installées correctement et que le fichier est exécuté dans le bon environnement.

---

### Auteur
Ce projet a été développé pour illustrer l'implémentation et l'utilisation des forêts aléatoires en Julia.
