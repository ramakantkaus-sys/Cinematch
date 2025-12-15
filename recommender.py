import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    """Loads the processed movie list."""
    try:
        movies_list = pickle.load(open('movies_list.pkl', 'rb'))
        return movies_list
    except FileNotFoundError:
        print("Error: 'movies_list.pkl' not found. Run data_preprocessing.py first.")
        return None

def train_model(movies_list):
    """Computes TF-IDF and Cosine Similarity."""
    print("Vectorizing text (TF-IDF)...")
    # TF-IDF Vectorizer
    # max_features=5000, stop_words='english' as requested
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    
    # Fit and transform
    vector = tfidf.fit_transform(movies_list['movie_profile']).toarray()
    
    print("Computing similarity matrix...")
    # Cosine Similarity
    similarity = cosine_similarity(vector)
    
    # Optimize size: Convert to float16 (halves size twice: 64 -> 32 -> 16)
    # 5000*5000 * 2 bytes = ~50MB, fitting easily in git limit.
    import numpy as np
    similarity = similarity.astype(np.float16)
    
    print("Saving similarity matrix...")
    pickle.dump(similarity, open('similarity.pkl', 'wb'))
    print("Model training complete. 'similarity.pkl' saved.")
    
    return similarity

def recommend(movie_title, movies_list, similarity):
    """
    Recommends movies based on similarity.
    Input: movie_title (str), movies_list (DataFrame), similarity (matrix)
    Output: List of recommended movie titles
    """
    # Find index
    try:
        movie_index = movies_list[movies_list['title'] == movie_title].index[0]
    except IndexError:
        return ["Movie not found in database."]
    
    # Get distances
    distances = similarity[movie_index]
    
    # Sort
    # enumerate to keep index, sort by score (x[1]) descending, take 1:6 (top 5 excluding self)
    movies_list_sorted = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_movies = []
    for i in movies_list_sorted:
        recommended_movies.append(movies_list.iloc[i[0]].title)
        
    return recommended_movies

if __name__ == '__main__':
    # Script mode: Train and save model
    data = load_data()
    if data is not None:
        train_model(data)
