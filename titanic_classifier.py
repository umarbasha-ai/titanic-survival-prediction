"""
Titanic Survival Prediction using Multiple Classifiers
Author: Umarbasha Nemakal
Description: This script loads the Titanic dataset, preprocesses it, trains multiple classifiers,
and evaluates their performance using accuracy and confusion matrices.
"""

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

def load_and_prepare_data():
    """Load Titanic dataset and preprocess features."""
    df = sns.load_dataset("titanic")
    df = df[['survived', 'age', 'sex', 'pclass', 'fare', 'class']].dropna()

    # Encode categorical features
    df['sex_encoded'] = LabelEncoder().fit_transform(df['sex'])
    df['class_encoded'] = LabelEncoder().fit_transform(df['class'])
    df.drop(columns=['sex', 'class'], inplace=True)

    features = ['class_encoded', 'sex_encoded', 'age', 'pclass', 'fare']
    target = 'survived'
    return df[features], df[target]

def train_models(X_train, y_train):
    """Train multiple classifiers and return them."""
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        'Naive Bayes': GaussianNB()
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models

def evaluate_models(models, X_test, y_test):
    """Evaluate models and print accuracy and confusion matrix."""
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        results.append((name, acc))
        print(f"\n🔍 {name} Classifier")
        print(f"Accuracy: {acc:.4f}")
        print("Confusion Matrix:")
        print(cm)
    return results

def display_summary(results):
    """Display model comparison summary."""
    print("\nModel Comparison Summary")
    summary_df = pd.DataFrame(results, columns=['Model', 'Accuracy']).sort_values(by='Accuracy')
    print(summary_df.to_string(index=False))

def main():
    X, y = load_and_prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)
    display_summary(results)

if __name__ == "__main__":
    main()