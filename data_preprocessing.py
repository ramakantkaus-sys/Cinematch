import pandas as pd
import ast
import pickle
import os

def load_data():
    """Loads the movies and credits datasets."""
    print("Loading datasets...")
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')
    return movies, credits

def merge_data(movies, credits):
    """Merges movies and credits on ID."""
    print("Merging datasets...")
    # Merge on data
    # movies.id is the same as credits.movie_id
    movies = movies.merge(credits, on='title')
    
    # After merge, we might have duplicate columns or need to fit the schema
    # The prompt explicitly asked to use movies.id == credits.movie_id
    # But usually merge is easier on 'title' if unique, or we can rename and merge.
    # Let's inspect the shapes in real run, but here I will trust the standard TMDB structure.
    # Actually, let's do it safely as requested:
    
    # However, since I merged on title above, let's fix it to be robust:
    # Reloading to follow strict prompt instructions:
    # "Use: movies.id == credits.movie_id"
    # The pandas way is merge left_on='id' right_on='movie_id'
    
    return movies

def clean_data(movies, credits):
    # Re-loading logic inside here to be cleaner if I were refactoring, 
    # but let's stick to the flow.
    
    # Correct merge strategy per instructions:
    movies = movies.merge(credits, left_on='id', right_on='movie_id')
    
    print("Selecting columns...")
    # Keep specific columns
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    
    print("Handling missing values...")
    movies.dropna(subset=['overview'], inplace=True)
    movies.fillna('', inplace=True)
    
    return movies

def parse_json_columns(movies):
    print("Parsing JSON columns...")
    
    def convert(obj):
        L = []
        try:
            for i in ast.literal_eval(obj):
                L.append(i['name'])
        except (ValueError, TypeError):
            pass
        return L

    def convert3(obj):
        L = []
        counter = 0
        try:
            for i in ast.literal_eval(obj):
                if counter != 3:
                    L.append(i['name'])
                    counter += 1
                else:
                    break
        except (ValueError, TypeError):
            pass
        return L

    def fetch_director(obj):
        L = []
        try:
            for i in ast.literal_eval(obj):
                if i['job'] == 'Director':
                    L.append(i['name'])
                    break
        except (ValueError, TypeError):
            pass
        return L

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert3)
    movies['crew'] = movies['crew'].apply(fetch_director) # This stores director in 'crew' column
    
    # Rename crew to director for clarity if we want, or just keep as crew but containing director list
    # The prompt says "Crew -> extract ONLY the director's name"
    # Let's rename the column for clarity in the profile
    movies['director'] = movies['crew']
    # movies = movies.drop(columns=['crew']) # Don't strictly need to drop, but cleaner
    
    return movies

def feature_engineering(movies):
    print("Feature engineering...")
    
    # Collapse spaces
    def collapse(L):
        L1 = []
        for i in L:
            L1.append(i.replace(" ","").lower())
        return L1
    
    movies['genres'] = movies['genres'].apply(collapse)
    movies['keywords'] = movies['keywords'].apply(collapse)
    movies['cast'] = movies['cast'].apply(collapse)
    movies['director'] = movies['director'].apply(collapse)
    
    # Overview is a string, let's split it to list to concatenate easily
    movies['overview'] = movies['overview'].apply(lambda x: x.split())
    
    # Create tags
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['director']
    
    # Convert tags back to string
    movies['movie_profile'] = movies['tags'].apply(lambda x: " ".join(x))
    
    # Final dataframe
    new_df = movies[['movie_id', 'title', 'movie_profile']]
    
    # The prompt asked for lowercase conversion and cleaning.
    # The collapse function did lowercase for lists.
    # The overview was split (implicitly handles some spaces) and joined.
    # Let's ensure final profile is lower case (it should be mostly, but overview words might not have been lowercased in the lambda x: x.split() step above unless I change it)
    
    # Fix overview lowercasing
    # Better approach:
    # The prompt says "Combine: overview, genres... Apply: lowercase conversion"
    # I'll do a final .lower() on the joined string to be safe and efficient.
    
    new_df['movie_profile'] = new_df['movie_profile'].apply(lambda x: x.lower())
    
    return new_df

def main():
    movies, credits = load_data()
    
    # Merge and clean
    # Note: my clean_data function actually did the merge inside it in the corrected logic.
    # Let's clean up the flow.
    
    # Correct Flow:
    # 1. Merge
    # movies and credits both have 'title' column, so they will be suffixed.
    movies_merged = movies.merge(credits, left_on='id', right_on='movie_id')
    
    # 2. Select Columns
    # 'title' becomes 'title_x' (from movies) and 'title_y' (from credits)
    # We'll use 'title_x' as the main title.
    movies_merged = movies_merged.rename(columns={'title_x': 'title'})
    movies_selected = movies_merged[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    
    # 3. Clean
    movies_selected.dropna(subset=['overview'], inplace=True)
    # create a copy to avoid SettingWithCopyWarning
    movies_selected = movies_selected.copy() 
    
    # 4. JSON Parse
    movies_parsed = parse_json_columns(movies_selected)
    
    # 5. Feature Engineering
    final_df = feature_engineering(movies_parsed)
    
    print("Saving processed data...")
    pickle.dump(final_df, open('movies_list.pkl', 'wb'))
    print("Done! 'movies_list.pkl' created.")
    
    # Also save the raw DataFrame if needed, but the prompt implies we build the recommender on the profile.
    # The recommender needs to return titles, which final_df has.
    
if __name__ == '__main__':
    main()
