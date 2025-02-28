# amazon_train.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
from scipy.stats import randint
from sklearn.metrics import r2_score


# Load datasets
credits_path = r"C:\Users\nag15\OneDrive\Desktop\Labmentix\Amazon_ML_Project\credits.csv"
titles_path = r"C:\Users\nag15\OneDrive\Desktop\Labmentix\Amazon_ML_Project\titles.csv"

def load_data(credits_path, titles_path):
    df1 = pd.read_csv(credits_path)
    df2 = pd.read_csv(titles_path)
    df = pd.merge(df1, df2, on='id', how='inner')
    return df

df = load_data(credits_path, titles_path)

# Handle missing values
def handle_missing_values(df):
    df.fillna({
        'character': 'Unknown',
        'description': 'No Description',
        'imdb_id': 'No IMDb ID',
        'imdb_score': df['imdb_score'].median(),
        'imdb_votes': df['imdb_votes'].median(),
        'tmdb_popularity': df['tmdb_popularity'].median(),
        'tmdb_score': df['tmdb_score'].median()
    }, inplace=True)
    return df

df = handle_missing_values(df)

# Convert genre and country columns into lists
df['genres'] = df['genres'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
df['production_countries'] = df['production_countries'].apply(lambda x: x.split(',') if isinstance(x, str) else [])

# Feature Engineering
df['genre_count'] = df['genres'].apply(len)
df['country_count'] = df['production_countries'].apply(len)
df['is_movie'] = (df['type'] == 'MOVIE').astype(int)

# Selecting Features and Target (Age Certification Removed)
features = ['runtime', 'tmdb_popularity', 'tmdb_score', 'genre_count', 'country_count', 'is_movie']
target = 'imdb_score'

X = df[features]
y = df[target]

# Correlation Heatmap (excluding 'id' column)
numeric_df = df.drop('id', axis=1).select_dtypes(include=np.number) #create dataframe that only contain numbers, and drops the id column.
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Feature Distribution Visualization
for feature in features:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[feature], kde=True)
    plt.title(f"Distribution of {feature}")
    plt.show()

# Splitting the Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training (Random Forest Regressor)
model = RandomForestRegressor(random_state=42)

# Hyperparameter Tuning (RandomizedSearchCV)
param_dist = {
    'n_estimators': randint(10, 50),
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 5)
}

grid_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, cv=3, n_iter=50, n_jobs=-1, verbose=2, random_state=42, scoring='r2')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Saving the Model
model_path = r"C:\Users\nag15\OneDrive\Desktop\Labmentix\Amazon_ML_Project\models\best_model.pkl"
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump((best_model, X.columns), model_path)
print(f"Best model saved at {model_path}")

# Overfitting Check (Training Score)
y_train_pred = best_model.predict(X_train)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, best_model.predict(X_test))
print(f"Training R² Score: {train_r2:.4f}")
print(f"Testing R^2 Score: {test_r2:.4f}")

if train_r2 - test_r2 > 0.1:
    print("Potential Overfitting Detected.")
else:
    print("Model appears to generalize well.")

# Feature Importance Analysis
importances = best_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 5))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title("Feature Importance - Random Forest")
plt.show()

# Residual Analysis
plt.figure(figsize=(10, 6))
sns.histplot(y_test - best_model.predict(X_test), bins=30, kde=True, color="purple")
plt.title("Residual Distribution")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.show() 