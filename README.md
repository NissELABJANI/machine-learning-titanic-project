# 🚢 Machine Learning Titanic Project

This project is a mini data science and machine learning analysis based on the Titanic dataset. The goal is to predict the survival of passengers using features such as age, sex, passenger class, and more.

The analysis includes data visualization, preprocessing, and training different ML models (logistic regression, random forest).

---

## 📌 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Screenshots](#screenshots)
- [Author](#author)
- [License](#license)

---

## 📊 Features

- Data loading and cleaning (handling nulls, encoding, etc.)
- Exploratory data analysis with `matplotlib` and `seaborn`
- Model training with `LogisticRegression` and `RandomForestClassifier`
- Evaluation using confusion matrices
- Survival predictions based on test input

---

## 📁 Project Structure

machine-learning-titanic-project/ ├── titanic_project_final.py # Main Python script ├── requirements.txt # Project dependencies ├── .gitignore # Git ignore config ├── README.md # Project documentation └── screenshots/ # Output plots ├── age_par_classe.png ├── confusion_logistique.png ├── confusion_random_forest.png ├── survie_par_classe.png └── survie_par_sexe.png

---

## ▶️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/NissELABJANI/machine-learning-titanic-project.git
cd machine-learning-titanic-project
```
### 2. Create and activate a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate    # or venv\Scripts\activate on Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the project

```bash
python titanic_project_final.py
```

## 🖼️ Screenshots
📈 Age par classe

🧮 Matrice de confusion (régression logistique)

🌲 Matrice de confusion (random forest)

📊 Survie par classe

👩‍🦱 Survie par sexe

## 👩‍💻 Author
Nissrine Elabjani
2nd Year Engineering Student – Embedded Systems
Université Paris Cité – Denis Diderot Engineering School
📍 Paris, France
📧 nissrine.elabjani@gmail.com

## 📄 License
This project is licensed under the MIT License. You are free to use, modify, and distribute this code with attribution.

