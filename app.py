import pickle
import streamlit as st
import pandas as pd
import requests

def fetch_poster(movie_id):
    """Fetches the movie poster from The Movie Database (TMDB) API."""
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
        data = requests.get(url)
        data.raise_for_status()  # Raise an exception for bad status codes
        data = data.json()
        poster_path = data['poster_path']
        full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
        return full_path
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return "https://placehold.co/500x750/333333/FFFFFF?text=No+Poster"
    except (KeyError, TypeError):
        st.error("Could not parse poster path from API response.")
        return "https://placehold.co/500x750/333333/FFFFFF?text=No+Poster"


def recommend(movie):
    """Recommends 5 similar movies based on the selected movie."""
    index = movies[movies['title'] == movie].index[0]
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

# --- Load Data ---
# Use st.cache_data to load data only once
@st.cache_data
def load_data():
    try:
        # Corrected the filename from 'movie_list.pkl' to 'movie.pkl'
        movies_dict = pickle.load(open('movie.pkl', 'rb'))
        movies_df = pd.DataFrame(movies_dict)
        similarity_matrix = pickle.load(open('similarity.pkl', 'rb'))
        return movies_df, similarity_matrix
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'movie.pkl' and 'similarity.pkl' are in the root directory.")
        return None, None

movies, similarity = load_data()

if movies is not None and similarity is not None:
    movie_list = movies['title'].values
    selected_movie = st.selectbox(
        "Type or select a movie from the dropdown",
        movie_list
    )

    if st.button('Show Recommendation'):
        with st.spinner('Fetching recommendations...'):
            recommended_movie_names, recommended_movie_posters = recommend(selected_movie)
            
            # Display recommendations in columns
            cols = st.columns(5)
            for i in range(5):
                with cols[i]:
                    st.text(recommended_movie_names[i])
                    st.image(recommended_movie_posters[i])
