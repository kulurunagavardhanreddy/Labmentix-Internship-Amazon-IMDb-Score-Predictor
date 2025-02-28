# amazon.py

import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Load the pre-trained model
model_path = r"C:\Users\nag15\OneDrive\Desktop\Labmentix\Amazon_ML_Project\best_model.pkl"
loaded_model, expected_features = joblib.load(model_path)

# Load the full dataframe for actual score retrieval
credits_path = r"C:\Users\nag15\Downloads\credits.csv"
titles_path = r"C:\Users\nag15\Downloads\titles.csv"

def load_data(credits_path, titles_path):
    df1 = pd.read_csv(credits_path)
    df2 = pd.read_csv(titles_path)
    df = pd.merge(df1, df2, on='id', how='inner')
    return df

df = load_data(credits_path, titles_path)

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
df['genres'] = df['genres'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
df['production_countries'] = df['production_countries'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
df['genre_count'] = df['genres'].apply(len)
df['country_count'] = df['production_countries'].apply(len)
df['is_movie'] = (df['type'] == 'MOVIE').astype(int)

# Streamlit App
st.title("Amazon Prime Movie IMDb Score Prediction")
st.write("Enter movie details to predict IMDb score:")

runtime = st.number_input("Runtime (minutes)", min_value=0, value=90)
tmdb_popularity = st.number_input("TMDB Popularity", min_value=0.0, value=5.0, format="%.3f")
tmdb_score = st.number_input("TMDB Score", min_value=0.0, max_value=10.0, value=7.0, format="%.2f")
genre_count = st.number_input("Number of Genres", min_value=0, value=2)
country_count = st.number_input("Number of Countries", min_value=0, value=1)
is_movie = st.selectbox("Type", ["Movie", "Show"])
is_movie = 1 if is_movie == "Movie" else 0

input_data = {
    'runtime': runtime,
    'tmdb_popularity': tmdb_popularity,
    'tmdb_score': tmdb_score,
    'genre_count': genre_count,
    'country_count': country_count,
    'is_movie': is_movie,
}

input_df = pd.DataFrame([input_data])
input_df = input_df[expected_features]

if st.button("Predict IMDb Score"):
    prediction = loaded_model.predict(input_df)
    predicted_score = prediction[0]
    st.write(f"Predicted IMDb Score: {predicted_score:.2f}")

    # Find matching row and display actual IMDb score
    tolerance = 1e-3 # Define a tolerance for floating-point comparisons
    matching_rows = df[
        (abs(df['runtime'] - runtime) < tolerance) &
        (abs(df['tmdb_popularity'] - tmdb_popularity) < tolerance) &
        (abs(df['tmdb_score'] - tmdb_score) < tolerance) &
        (df['genre_count'] == genre_count) &
        (df['country_count'] == country_count) &
        (df['is_movie'] == is_movie)
    ]

    if not matching_rows.empty:
        actual_score = matching_rows['imdb_score'].iloc[0]
        st.write(f"Actual IMDb Score: {actual_score:.2f}")
        st.subheader("Matching Movie Data:")
        st.dataframe(matching_rows)
    else:
        st.write("No matching movie found in the dataset.")
        
    # Accuracy and Actual vs. Predicted (using test data from build_model.py)
    features = ['runtime', 'tmdb_popularity', 'tmdb_score', 'genre_count', 'country_count', 'is_movie']
    target = 'imdb_score'

    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred_test = loaded_model.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    st.write(f"Model R² Score (on test data): {r2:.4f}")

    st.subheader("Actual vs. Predicted IMDb Scores (Sample)")
    sample_df = pd.DataFrame({'Actual IMDb': y_test, 'Predicted IMDb': y_pred_test})
    st.dataframe(sample_df.head(10))