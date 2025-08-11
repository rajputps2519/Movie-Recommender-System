# Movie Recommendation System

This is a **Content-Based Movie Recommendation System** that suggests movies similar to a user's choice. The project leverages natural language processing techniques to analyze movie attributes like plot, genre, keywords, cast, and crew. The recommendation engine is deployed as a user-friendly web application using Streamlit.

The core of the system is built on creating a "tag" for each movie by combining its most important textual data. These tags are then converted into numerical vectors using **TF-IDF (Term Frequency-Inverse Document Frequency)**, and the **cosine similarity** between these vectors is calculated to determine how "similar" two movies are.

---
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://movie-recommender-system-prs.streamlit.app/)
## 🚀 Key Features

-   **Content-Based Filtering:** Recommends movies by analyzing and comparing their content attributes.
-   **Data Preprocessing:** Cleans and merges data from two datasets (`movies.csv` and `credits.csv`) to create a unified and usable dataset.
-   **Feature Engineering:** Combines genres, keywords, plot overview, cast (top 3 actors), and director into a single text "tag" for each movie.
-   **Vectorization:** Uses `TfidfVectorizer` to transform text tags into a high-dimensional vector space.
-   **Similarity Score:** Employs `cosine_similarity` to measure the similarity between movies efficiently.
-   **Interactive Web App:** A simple and intuitive UI built with **Streamlit** allows users to select a movie and instantly receive five recommendations, complete with posters.

---
## 🛠️ Technology Stack

-   **Programming Language:** Python
-   **Data Manipulation:** Pandas, NumPy
-   **Machine Learning:** Scikit-learn
-   **Web Framework:** Streamlit
-   **Data Processing:** AST (for parsing string-formatted lists)
-   **API Integration:** Requests (to fetch movie posters from The Movie Database API)

---
## 📂 Project Structure

├── .ipynb_checkpoints/   # Jupyter Notebook checkpoints
├── app.py                # Main Streamlit application script
├── movie_recommender.ipynb # Jupyter Notebook for model development and data processing
├── movies.csv            # Dataset containing movie information
├── credits.csv           # Dataset containing cast and crew information
├── movie_list.pkl        # Serialized pandas DataFrame with preprocessed movie data
├── similarity.pkl        # Serialized cosine similarity matrix
└── requirements.txt      # List of Python dependencies
---
## ⚙️ How It Works

1.  **Data Loading & Merging:** The `movies.csv` and `credits.csv` datasets are loaded into pandas DataFrames and merged based on the movie title.
2.  **Data Cleaning & Preprocessing:**
    -   Irrelevant columns are dropped.
    -   Missing values are handled.
    -   JSON-like string columns (genres, keywords, cast, crew) are parsed to extract relevant information (e.g., genre names, top 3 actors, director).
3.  **Feature Combination:** The extracted features (overview, genres, keywords, cast, and crew) are concatenated into a single string, or "tag," for each movie.
4.  **Text Vectorization:** The collection of all movie tags is transformed into a matrix of TF-IDF features. This process gives more weight to words that are significant to a specific movie while down-weighting common words.
5.  **Similarity Calculation:** The cosine similarity is computed between all pairs of movie vectors. The resulting matrix gives a similarity score from 0 to 1, where 1 means the movies are identical in content.
6.  **Recommendation Function:** When a user selects a movie, the system finds its index, retrieves its similarity scores with all other movies, sorts them in descending order, and returns the top 5 most similar movies.
7.  **Web Interface:** The Streamlit app loads the pre-processed movie list and the similarity matrix. It provides a dropdown for the user to select a movie and displays the recommended movie titles and their posters fetched via the TMDB API.

---
## 📦 Installation & Usage

To run this project on your local machine, follow these steps.

### 1. Clone the Repository
```bash
git clone [https://github.com/rajputps2519/Movie-Recommender-System.git](https://github.com/rajputps2519/Movie-Recommender-System.git)
cd Movie-Recommender-System

Install Dependencies
Install all the required Python libraries from the requirements.txt file.
pip install -r requirements.
3. Get a TMDB API Key
This project requires an API key from The Movie Database (TMDB) to fetch movie posters.

Create a free account at TMDB.

Go to your account Settings -> API and request an API key.

Note: The current implementation in app.py does not require you to input the key manually, but it's good practice to manage keys securely if you modify the code.
4. Run the Streamlit App
streamlit run app.py
The application will open in your default web browser, where you can select a movie to get recommendations.

