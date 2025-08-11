Movie Recommendation System
This project is a Content-Based Movie Recommendation System built using Python, Pandas, and Scikit-learn. It recommends movies by analyzing shared attributes like genre, keywords, cast, and crew. The user interface is a web application powered by Streamlit.
Features
Content-Based Filtering: Recommends movies based on content similarity rather than user ratings.

TF-IDF Vectorization: Converts text data (like plot summaries, genres, and keywords) into meaningful numerical vectors.

Cosine Similarity: Calculates the similarity between movies to find the best matches.

Interactive UI: A simple and user-friendly web application built with Streamlit allows users to easily select a movie and get recommendations.

Preprocessed Data: Utilizes pre-cleaned movie data and a pre-computed similarity matrix for fast and efficient recommendations.

Technology Stack
Python: Core programming language.

Pandas: For data manipulation and preprocessing.

Scikit-learn: For implementing TF-IDF Vectorization and Cosine Similarity.

Streamlit: To create the interactive web application.

Jupyter Notebook: For data analysis and model development (implied).
Project Structure
Here is the file structure of the project:
.
├── app.py              # Main Streamlit application script
├── movie_list.pkl      # Serialized pandas DataFrame with preprocessed movie data
├── similarity.pkl      # Serialized similarity matrix for movie recommendations
├── requirements.txt    # List of Python dependencies for the project
└── README.md           # Project documentation
 Installation & Usage
Follow these steps to get the project up and running on your local machine.
