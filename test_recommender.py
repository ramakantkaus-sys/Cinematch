import pickle
from recommender import recommend
import pandas as pd

def test_recommender():
    print("Loading data for testing...")
    try:
        movies_list = pickle.load(open('movies_list.pkl', 'rb'))
        similarity = pickle.load(open('similarity.pkl', 'rb'))
    except FileNotFoundError:
        print("FAIL: Data files not found.")
        return

    test_movie = "Avatar"
    print(f"Testing recommendation for: {test_movie}")
    
    # Check if movie exists
    if test_movie not in movies_list['title'].values:
        print(f"WARNING: {test_movie} not in dataset. Picking first movie: {movies_list.iloc[0].title}")
        test_movie = movies_list.iloc[0].title

    recs = recommend(test_movie, movies_list, similarity)
    print("Recommendations:", recs)
    
    if len(recs) == 5:
        print("PASS: Got 5 recommendations.")
    else:
        print(f"FAIL: Expected 5 recommendations, got {len(recs)}")

    # Test unknown movie
    unknown_recs = recommend("NonExistentMovie123", movies_list, similarity)
    if "Movie not found" in unknown_recs[0]:
        print("PASS: Error handling for unknown movie works.")
    else:
        print("FAIL: Unknown movie should return error.")

if __name__ == "__main__":
    test_recommender()
