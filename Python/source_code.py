
# 1. Importation des bibliothèques

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

# 2. Génération des données aléatoires"

# Définir le nombre d'entrées
n = 250
random.seed(42)

# Création du dataset
data = {
    "ID": [i + 1 for i in range(n)],
    "Age": [random.randint(10, 18) for _ in range(n)],
    "Sexe": [random.choice(["M", "F"]) for _ in range(n)],
    "Pref_Visuel": [],
    "Pref_Auditif": [],
    "Pref_Kinesthesique": [],
    "Math_Score": [],
    "Sciences_Score": [],
    "Langues_Score": [],
    "Temps_Etude_Visuel": [],
    "Temps_Etude_Auditif": [],
    "Temps_Etude_Kinesthesique": [],
    }

# Classes bien définies : Visuel, Auditif, Kinesthésique, Mixte
for _ in range(n):
    apprenant_type = random.choice(["Visuel", "Auditif", "Kinesthesique", "Mixte"])

    if apprenant_type == "Visuel":
        pref_visuel = random.randint(8, 10)
        pref_auditif = random.randint(0, 2)
        pref_kinesthesique = random.randint(0, 2)
        temps_etude_visuel = random.uniform(7, 10)
        temps_etude_auditif = random.uniform(1, 3)
        temps_etude_kinesthesique = random.uniform(1, 3)

    elif apprenant_type == "Auditif":
        pref_visuel = random.randint(0, 2)
        pref_auditif = random.randint(8, 10)
        pref_kinesthesique = random.randint(0, 2)
        temps_etude_visuel = random.uniform(1, 3)
        temps_etude_auditif = random.uniform(7, 10)
        temps_etude_kinesthesique = random.uniform(1, 3)

    elif apprenant_type == "Kinesthesique":
        pref_visuel = random.randint(0, 2)
        pref_auditif = random.randint(0, 2)
        pref_kinesthesique = random.randint(8, 10)
        temps_etude_visuel = random.uniform(1, 3)
        temps_etude_auditif = random.uniform(1, 3)
        temps_etude_kinesthesique = random.uniform(7, 10)

    else:  # Mixte
        pref_visuel = random.randint(5, 8)
        pref_auditif = random.randint(5, 8)
        pref_kinesthesique = random.randint(5, 8)
        temps_etude_visuel = random.uniform(4, 7)
        temps_etude_auditif = random.uniform(4, 7)
        temps_etude_kinesthesique = random.uniform(4, 7)

    # Scores et satisfaction
    math_score = random.randint(50, 100)
    sciences_score = random.randint(50, 100)
    langues_score = random.randint(50, 100)


    # Ajout au dataset
    data["Pref_Visuel"].append(pref_visuel)
    data["Pref_Auditif"].append(pref_auditif)
    data["Pref_Kinesthesique"].append(pref_kinesthesique)
    data["Math_Score"].append(math_score)
    data["Sciences_Score"].append(sciences_score)
    data["Langues_Score"].append(langues_score)
    data["Temps_Etude_Visuel"].append(temps_etude_visuel)
    data["Temps_Etude_Auditif"].append(temps_etude_auditif)
    data["Temps_Etude_Kinesthesique"].append(temps_etude_kinesthesique)
   

# Conversion en DataFrame
df = pd.DataFrame(data)

# Déterminer le type d'apprenant
def determine_apprenant(row):
    preferences = {
        "Visuel": row["Pref_Visuel"],
        "Auditif": row["Pref_Auditif"],
        "Kinesthesique": row["Pref_Kinesthesique"],
    }
    max_pref = max(preferences, key=preferences.get)
    if list(preferences.values()).count(preferences[max_pref]) > 1:
        return "Mixte"
    return max_pref

# Ajouter la colonne "Apprenant_Type"
df["Apprenant_Type"] = df.apply(determine_apprenant, axis=1)

# Sauvegarder dans un fichier CSV
df.to_csv("apprenants_dataset.csv", index=False)

print("Dataset généré et sauvegardé sous 'apprenants_dataset.csv'")

"""# 3. Inspection et Visualisation des données"""

print("Statistiques descriptives :")
print(df.describe())
df.head()

# Affichage de la distribution de chaque caractéristique (score)
features = ['Math_Score', 'Sciences_Score', 'Langues_Score',
            'Temps_Etude_Visuel', 'Temps_Etude_Auditif',
            'Temps_Etude_Kinesthesique']

plt.figure(figsize=(15, 10))
for i, feature in enumerate(features, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution de {feature}')
plt.tight_layout()
plt.show()

# Analyse de corrélation entre les caractéristiques
correlation_matrix = df[features].corr()

# Plotting the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matrice de Corrélation des Caractéristiques')
plt.show()

# Visualisation des types d'apprenants
plt.figure(figsize=(6, 4))
sns.countplot(x='Apprenant_Type', data=df, palette="Set2")
plt.title('Distribution des Types d\'Apprenants')
plt.show()

# Boxplot des scores par type d'apprenant
plt.figure(figsize=(12, 6))
sns.boxplot(x='Apprenant_Type', y='Math_Score', data=df, palette="Set2")
plt.title('Distribution des scores de Math par type d\'apprenant')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Apprenant_Type', y='Sciences_Score', data=df, palette="Set2")
plt.title('Distribution des scores de Sciences par type d\'apprenant')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Apprenant_Type', y='Langues_Score', data=df, palette="Set2")
plt.title('Distribution des scores de Langues par type d\'apprenant')
plt.show()

# Analyser la corrélation entre le temps d'étude et les scores
study_time_columns = ['Temps_Etude_Visuel', 'Temps_Etude_Auditif', 'Temps_Etude_Kinesthesique']
performance_columns = ['Math_Score', 'Sciences_Score', 'Langues_Score']

# Créer une matrice de corrélation pour le temps d'étude et les performances
study_performance_corr = df[study_time_columns + performance_columns].corr()

# 4. Code du modèle : Random Forest

# --- Classes pour Random Forest et Decision Tree ---
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([self._predict_row(row, self.tree) for _, row in X.iterrows()])

    def _build_tree(self, X, y, depth):
        if len(set(y)) == 1 or (self.max_depth and depth >= self.max_depth):
            return Counter(y).most_common(1)[0][0]

        best_feature, best_threshold = self._find_best_split(X, y)
        if best_feature is None:
            return Counter(y).most_common(1)[0][0]

        left_indices = X[best_feature] <= best_threshold
        right_indices = ~left_indices
        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {"feature": best_feature, "threshold": best_threshold, "left": left_tree, "right": right_tree}

    def _find_best_split(self, X, y):
        best_feature, best_threshold = None, None
        best_gain = -1

        for feature in X.columns:
            thresholds = X[feature].unique()
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature, threshold)
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, feature, threshold
        return best_feature, best_threshold

    def _information_gain(self, X, y, feature, threshold):
        left_y = y[X[feature] <= threshold]
        right_y = y[X[feature] > threshold]
        if len(left_y) == 0 or len(right_y) == 0:
            return 0

        total_entropy = self._entropy(y)
        left_entropy = self._entropy(left_y)
        right_entropy = self._entropy(right_y)

        left_weight = len(left_y) / len(y)
        right_weight = len(right_y) / len(y)
        return total_entropy - (left_weight * left_entropy + right_weight * right_entropy)

    @staticmethod
    def _entropy(y):
        counts = Counter(y)
        probabilities = [count / len(y) for count in counts.values()]
        return -sum(p * np.log2(p) for p in probabilities)

    def _predict_row(self, row, tree):
        if not isinstance(tree, dict):
            return tree
        if row[tree["feature"]] <= tree["threshold"]:
            return self._predict_row(row, tree["left"])
        else:
            return self._predict_row(row, tree["right"])

    def visualize_tree(self, tree=None, depth=0):
        if tree is None:
            if self.tree is None:
                print("L'arbre n'a pas été entraîné.")
                return
            tree = self.tree
        if not isinstance(tree, dict):
            print("\t" * depth + f"Predict: {tree}")
            return
        print("\t" * depth + f"[Feature: {tree['feature']}, Threshold: {tree['threshold']}]")
        self.visualize_tree(tree["left"], depth + 1)
        self.visualize_tree(tree["right"], depth + 1)

class RandomForest:
    def __init__(self, n_trees=10, max_depth=None, sample_size=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.sample_size = sample_size
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            sample_indices = np.random.choice(len(X), size=self.sample_size or len(X), replace=True)
            X_sample, y_sample = X.iloc[sample_indices], y.iloc[sample_indices]
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=tree_predictions)

# 5. Entraînement du modèle

# Chargement des données
df = pd.read_csv("apprenants_dataset.csv")

# Encodage des colonnes catégoriques (Sexe, Apprenant_Type)
df["Sexe"] = df["Sexe"].map({"M": 0, "F": 1})
le = LabelEncoder()
df["Apprenant_Type"] = le.fit_transform(df["Apprenant_Type"])

# Définir X (features) et y (target)
X = df.drop(columns=["ID", "Apprenant_Type"])
y = df["Apprenant_Type"]

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Entraînement et Visualisation ---
forest = RandomForest(n_trees=5, max_depth=5, sample_size=100)
forest.fit(X_train, y_train)

# Visualiser les arbres
for i, tree in enumerate(forest.trees):
    print(f"\n--- Arbre {i + 1} ---")
    tree.visualize_tree()

y_pred = forest.predict(X_test)

"""# 6. Évaluation du modèle"""

# Modèle (avec les meilleurs hyperparamètres : max_depth: 10, min_samples_split: 5, n_estimators: 50)
model = RandomForest(n_trees=50, max_depth=10, sample_size=50)
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Évaluer le modèle
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle : {accuracy:.2f}")

# --- Matrice de confusion ---
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Matrice de Confusion")
plt.xlabel("Prédictions")
plt.ylabel("Vérités")
plt.show()

# --- Calcul des importances des caractéristiques ---
def calculate_feature_importances(forest, X):
    feature_importances = np.zeros(len(X.columns))
    for tree in forest.trees:
        if hasattr(tree, "tree") and isinstance(tree.tree, dict):
            _accumulate_feature_importances(tree.tree, feature_importances, X.columns)
    return feature_importances / np.sum(feature_importances)

def _accumulate_feature_importances(node, importances, feature_names):
    if isinstance(node, dict):
        feature_index = pd.Index(feature_names).get_loc(node["feature"])  # Utilisation de l'index réel
        importances[feature_index] += 1  # Accumulation des splits pour l'importance
        _accumulate_feature_importances(node["left"], importances, feature_names)
        _accumulate_feature_importances(node["right"], importances, feature_names)

# Calcul des importances
feature_importances = calculate_feature_importances(model, X)

# Visualisation des importances
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=X.columns, palette="viridis")
plt.title("Importance des Caractéristiques")
plt.xlabel("Importance")
plt.ylabel("Caractéristique")
plt.show()

# --- Rapport de classification ---
report = classification_report(y_test, y_pred, output_dict=True, target_names=le.classes_)
report_df = pd.DataFrame(report).transpose()

# Supprimer 'accuracy' et 'macro avg' s'ils existent
if 'accuracy' in report_df.index:
    report_df = report_df.drop('accuracy')
if 'macro avg' in report_df.index:
    report_df = report_df.drop('macro avg')

plt.figure(figsize=(10, 6))
sns.heatmap(report_df.iloc[:, :-1], annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Rapport de Classification")
plt.xlabel("Métriques")
plt.ylabel("Classes")
plt.show()

"""# Réduction de Dimensionnalité avec la PCA pour Visualisation 2D et Identification des Groupes"""

# Charger le dataset
df = pd.read_csv("apprenants_dataset.csv")

# Sélectionner les caractéristiques
X = df[['Pref_Visuel', 'Pref_Auditif', 'Pref_Kinesthesique',
        'Math_Score', 'Sciences_Score', 'Langues_Score',
        'Temps_Etude_Visuel', 'Temps_Etude_Auditif', 'Temps_Etude_Kinesthesique'
        ]].values
y = df['Apprenant_Type']

# Encoder la variable cible
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Division train/test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Appliquer PCA pour réduire à 2 dimensions
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Entraîner votre modèle personnalisé
model = RandomForest(n_trees=10, max_depth=5, sample_size=100)
model.fit(pd.DataFrame(X_train_pca), pd.Series(y_train))

# Fonction pour tracer la frontière de décision
def plot_decision_boundary(X, y, model, feature_names, le):
    cmap_background = ListedColormap(['#FFAAAA', '#AAAAFF', '#AAFFAA', '#FFD700'])
    cmap_points = ListedColormap(['#FF0000', '#0000FF', '#00FF00', '#FFC000'])

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z = model.predict(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(12, 6))  # Réduire la taille du graphique
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_background)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_points, edgecolor='k')

    # Ajouter une légende en dehors de la figure
    handles = scatter.legend_elements()[0]
    labels = le.classes_
    plt.legend(handles, labels, title="Types d'apprenants", loc="center left", bbox_to_anchor=(1, 0.5))

    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title("Frontière de décision du Random Forest après PCA")
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.show()

# Tracer la frontière de décision après PCA avec votre modèle
plot_decision_boundary(X_train_pca, y_train, model, ['PCA1', 'PCA2'], le)

"""# 7. Prédiction de la classe d'un étudiant"""

# Charger le dataset
df = pd.read_csv("apprenants_dataset.csv")

# Encoder les colonnes catégoriques
df['Sexe_encoded'] = df['Sexe'].map({'M': 0, 'F': 1})

# Définir les caractéristiques (features) et la cible (target)
features = [
    'Pref_Visuel', 'Pref_Auditif', 'Pref_Kinesthesique',
    'Math_Score', 'Sciences_Score', 'Langues_Score',
    'Temps_Etude_Visuel', 'Temps_Etude_Auditif', 'Temps_Etude_Kinesthesique',
     'Sexe_encoded'
]
X = df[features]
y = df['Apprenant_Type']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner un modèle RandomForest
rf = RandomForest(n_trees=10, max_depth=5, sample_size=100)
rf.fit(X_train, y_train)

# Fonction pour prédire la classe d'un étudiant
def predict_student_class(model, student_data):
 
    required_features = [
        'Pref_Visuel', 'Pref_Auditif', 'Pref_Kinesthesique',
        'Math_Score', 'Sciences_Score', 'Langues_Score',
        'Temps_Etude_Visuel', 'Temps_Etude_Auditif', 'Temps_Etude_Kinesthesique',
        'Sexe_encoded'
    ]
    
    for feature in required_features:
        if feature not in student_data:
            raise ValueError(f"La caractéristique '{feature}' est manquante dans les données de l'étudiant.")
    
    # Convertir les données en DataFrame
    student_df = pd.DataFrame([student_data])
    
    # Prédire la classe
    predicted_class = model.predict(student_df)[0]
    
    return predicted_class

# Exemple de prédiction pour un étudiant
new_student = {
    "Pref_Visuel": 8,
    "Pref_Auditif": 6,
    "Pref_Kinesthesique": 7,
    "Math_Score": 85,
    "Sciences_Score": 78,
    "Langues_Score": 92,
    "Temps_Etude_Visuel": 5.5,
    "Temps_Etude_Auditif": 6.0,
    "Temps_Etude_Kinesthesique": 7.5,
    "Sexe_encoded": 1  # 0: M, 1: F
}

try:
    predicted_class = predict_student_class(rf, new_student)
    print(f"Classe prédite pour l'étudiant : {predicted_class}")
except ValueError as e:
    print(e)