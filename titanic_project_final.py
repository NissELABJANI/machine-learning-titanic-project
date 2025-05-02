# titanic_project_final.py 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import fetch_openml

# Charger les données depuis OpenML
titanic = fetch_openml(name="titanic", version=1, as_frame=True)
df = titanic.frame

# ----- Exploration des données -----
print(df.info())
print(df.describe())

# Survie selon la classe
sns.countplot(x="pclass", hue="survived", data=df)
plt.title("Survie selon la classe")
plt.savefig("survie_par_classe.png")
plt.clf()

# Survie selon le sexe
sns.countplot(x="sex", hue="survived", data=df)
plt.title("Survie selon le sexe")
plt.savefig("survie_par_sexe.png")
plt.clf()

# Âge par classe
sns.boxplot(x="pclass", y="age", data=df)
plt.title("Répartition de l'âge selon la classe")
plt.savefig("age_par_classe.png")
plt.clf()

# ----- Prétraitement -----
df['age'] = df['age'].fillna(df['age'].median())
df['fare'] = df['fare'].fillna(df['fare'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['embarked'], drop_first=True)

# Sélection des colonnes pertinentes
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare'] + [col for col in df.columns if col.startswith('embarked_')]
X = df[features].copy()

# Vérifier et corriger les NaN restants
X = X.fillna(X.median(numeric_only=True))
y = df['survived'].astype(int)

# ----- Modélisation : Régression Logistique -----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"[Régression Logistique] Accuracy: {accuracy:.2f}")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Matrice de confusion - Régression Logistique")
plt.savefig("confusion_logistique.png")
plt.clf()

# ----- Modélisation bonus : Random Forest -----
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)
print(f"[Random Forest] Accuracy: {rf_acc:.2f}")

rf_cm = confusion_matrix(y_test, rf_pred)
rf_disp = ConfusionMatrixDisplay(confusion_matrix=rf_cm)
rf_disp.plot()
plt.title("Matrice de confusion - Random Forest")
plt.savefig("confusion_random_forest.png")
plt.clf()
