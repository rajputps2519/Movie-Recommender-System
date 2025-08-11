import pickle
import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def fetch_poster(movie_id):
    """Fetches the movie poster from The Movie Database (TMDB) API."""
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
        data = requests.get(url)
        data.raise_for_status()  # Raise an exception for bad status codes
        data = data.json()
        poster_path = data.get('poster_path')
        if poster_path:
            full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
            return full_path
    except requests.exceptions.RequestException as e:
        # Silently log the error for debugging, don't show to user
        # st.error(f"API request failed: {e}") 
        pass
    except (KeyError, TypeError):
        # Silently log the error
        pass
    # Return a placeholder if any part of the process fails
    return "https://placehold.co/500x750/333333/FFFFFF?text=Poster+Not+Available"


def recommend(movie):
    """Recommends 5 similar movies based on the selected movie."""
    try:
        index = movies[movies['title'] == movie].index[0]
    except IndexError:
        st.error("Movie not found in the dataset.")
        return [], []
        
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    
    recommended_movie_names = []
    recommended_movie_posters = []
    
    for i in distances[1:6]:
        # fetch the movie poster
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names, recommended_movie_posters

# --- App Layout ---
st.set_page_config(layout="wide")
st.header('Movie Recommendation System')

# --- Load Data and Compute Similarity ---
# This function now also calculates the similarity matrix, which can be slow.
@st.cache_data
def load_and_process_data():
    try:
        movies_df = pd.DataFrame(pickle.load(open('movie.pkl', 'rb')))
        
        # --- THIS IS THE SLOW PART ---
        # Instead of loading similarity.pkl, we calculate it here.
        cv = CountVectorizer(max_features=5000, stop_words='english')
        vectors = cv.fit_transform(movies_df['tag']).toarray()
        similarity_matrix = cosine_similarity(vectors)
        # -----------------------------

        return movies_df, similarity_matrix
    except FileNotFoundError:
        st.error("Model file 'movie.pkl' not found. Please ensure it is in the root directory.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while loading or processing the data: {e}")
        return None, None


movies, similarity = load_and_process_data()

if movies is not None and similarity is not None:
    movie_list = movies['title'].values
    selected_movie = st.selectbox(
        "Type or select a movie from the dropdown",
        movie_list
    )

    if st.button('Show Recommendation'):
        with st.spinner('Fetching recommendations...'):
            recommended_movie_names, recommended_movie_posters = recommend(selected_movie)
            
            if recommended_movie_names:
                # Display recommendations in columns
                cols = st.columns(5)
                for i in range(len(recommended_movie_names)):
                    with cols[i]:
                        st.text(recommended_movie_names[i])
                        st.image(recommended_movie_posters[i])
