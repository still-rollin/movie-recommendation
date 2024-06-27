import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Load datasets
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Display the first few rows of each dataframe
print("Movies DataFrame:")
print(movies.head())
print("\nRatings DataFrame:")
print(ratings.head())

# Merge ratings and movies dataframes on movieId
data = pd.merge(ratings, movies, on='movieId')

# Create a utility matrix
utility_matrix = data.pivot(index='userId', columns='title', values='rating').fillna(0)
print("\nUtility Matrix:")
print(utility_matrix.head())

# Convert the utility matrix to a sparse matrix format
utility_matrix_sparse = csr_matrix(utility_matrix.values)

# Build the k-nearest neighbors model
model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model.fit(utility_matrix_sparse)

# Function to make movie recommendations
def recommend_movies(user_id, num_recommendations):
    # Get the index of the user in the utility matrix
    user_index = utility_matrix.index.get_loc(user_id)
    
    # Get the user's ratings from the utility matrix
    user_ratings = utility_matrix_sparse[user_index]
    
    # Find the k-nearest neighbors of the user
    distances, indices = model.kneighbors(user_ratings, n_neighbors=num_recommendations + 1)
    
    # Get the indices of the recommended movies
    movie_indices = indices.flatten()[1:]
    
    # Get the movie titles from the indices
    recommended_movies = utility_matrix.columns[movie_indices]
    
    return recommended_movies

