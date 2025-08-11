import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movies data
movies_dict = pickle.load(open('movie.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

# Create the similarity matrix from tags
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tag']).toarray()
similarity = cosine_similarity(vectors)

# Function to recommend movies
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies

# Streamlit UI
st.title('🎬 Movie Recommender System')

selected_movie = st.selectbox(
    'Type or select a movie from the dropdown',
    movies['title'].values
)

if st.button('Show Recommendation'):
    recommendations = recommend(selected_movie)
    st.write("**Recommended Movies:**")
    for rec in recommendations:
        st.write(rec)
