import streamlit as st
import pickle
import pandas as pd
import base64
import random

# Import recommender logic
from recommender import recommend

# -------------------------------------------------------------------------------------------------
# FUNCTIONS
# -------------------------------------------------------------------------------------------------

def load_data():
    """Loads the precomputed movie list and similarity matrix."""
    movies_list = pickle.load(open('movies_list.pkl', 'rb'))
    similarity = pickle.load(open('similarity.pkl', 'rb'))
    return movies_list, similarity

def get_base64_of_bin_file(bin_file):
    """Reads a binary file and returns the base64 string."""
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

def set_random_background():
    """Sets a random background image from the local folder."""
    images = ["1.jpg", "2.jpg", "4.jpg", "3.avif"]
    random_image = random.choice(images)
    bin_str = get_base64_of_bin_file(random_image)
    
    if bin_str:
        page_bg_img = f'''
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bin_str}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}
        .block-container {{
            background-color: rgba(0, 0, 0, 0.85);
            border-radius: 15px;
            padding: 3rem;
            margin-top: 2rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        h1, h2, h3, h4, h5, h6, p, li, span, label, div {{
            color: #E0E0E0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        .stButton>button {{
            background-color: #ff4b4b;
            color: white;
            border-radius: 5px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: bold;
            transition: all 0.3s ease;
        }}
        .stButton>button:hover {{
            background-color: #ff2b2b;
            transform: scale(1.05);
        }}
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)

# -------------------------------------------------------------------------------------------------
# CONFIG & SETUP
# -------------------------------------------------------------------------------------------------

st.set_page_config(page_title="Cinematch - Movie Recommender", page_icon="🎬", layout="wide")

# Apply Random Background
set_random_background()

# Load Data
try:
    movies, similarity = load_data()
except FileNotFoundError:
    st.error("Critical Error: Data files not found. Please run the preprocessing scripts first.")
    st.stop()

# -------------------------------------------------------------------------------------------------
# MAIN UI
# -------------------------------------------------------------------------------------------------

st.title("🎬 Cinematch")
st.markdown("### Intelligent Movie Recommendation System")
st.write("Experience the power of AI-driven content discovery.")
st.markdown("---")

col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("🔍 Find a Movie")
    movie_list = movies['title'].values
    selected_movie = st.selectbox(
        "Start typing to search for a movie you enjoyed:",
        movie_list
    )

    if st.button('🚀 Get Recommendations', use_container_width=True):
        with st.spinner('Analyzing movie attributes...'):
            recommendations = recommend(selected_movie, movies, similarity)
        
        st.markdown(f"### 🎯 Top Picks for you:")
        for i, movie in enumerate(recommendations):
            st.success(f"**{i+1}. {movie}**")

with col2:
    st.markdown("### 📊 Project Statistics")
    st.info(
        f"""
        - **Dataset**: TMDB 5000 Movies
        - **Total Movies**: {len(movie_list)}
        - **Features Used**: Overviews, Genres, Keywords, Cast, Crew
        - **Algorithm**: TF-IDF & Cosine Similarity
        """
    )

# -------------------------------------------------------------------------------------------------
# DEEP DIVE SECTION
# -------------------------------------------------------------------------------------------------

st.markdown("---")
st.subheader("🛠️ Technical Deep Dive")
st.markdown("Explore the engineering behind the recommendation engine.")

with st.expander("Step 1: Data Processing Pipeline (ETL)"):
    st.markdown("""
    **Getting the Data Ready**
    1.  **Ingestion**: Loading raw data from `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`.
    2.  **Merging**: Joining the datasets on the unique Movie ID to combine metadata with cast/crew info.
    3.  **Cleaning**: Removing null values and parsing complex JSON columns (e.g., extracting 'Science Fiction' from `[{"id": 878, "name": "Science Fiction"}]`).
    4.  **Feature Engineering**: 
        - Created a unified **`movie_profile`** tag.
        - *Example*: For "Avatar", the profile combines *"In the 22nd century..."* (Overview) + *"Action, Adventure"* (Genres) + *"SamWorthington"* (Cast) + *"JamesCameron"* (Director).
    """)

with st.expander("Step 2: Vectorization (TF-IDF)"):
    st.markdown("""
    **Translating Text to Numbers**
    Computers understand numbers, not words. We use **TF-IDF (Term Frequency-Inverse Document Frequency)**.
    
    - **TF**: How often a word appears in a specific movie profile.
    - **IDF**: How rare the word is across *all* movies.
    
    *Why?* Words like "the" or "movie" are common and useless. Words like "Alien" or "Cameron" are rare and important. TF-IDF gives higher weight to these unique identifiers.
    """)

with st.expander("Step 3: Similarity Engine (Cosine Similarity)"):
    st.markdown("""
    **Finding the Match**
    We treat every movie as a point in a 5000-dimensional space.
    
    - We calculate the **Cosine Similarity** (the angle) between the vector of your selected movie and every other movie.
    - **Score 1.0**: Identical movies.
    - **Score 0.0**: Completely unrelated.
    
    The system sorts all movies by this score and returns the top 5 nearest neighbors.
    """)
