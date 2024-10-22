import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request, jsonify
from scipy.sparse import csr_matrix, vstack
import time
from joblib import Memory
from functools import lru_cache
import threading
from sklearn.neighbors import NearestNeighbors

# Initialize memory caching
memory = Memory(location='.cache', verbose=0)
app = Flask(__name__)

# Global variables for pre-computed data
global_data = threading.local()

@memory.cache
def load_data():
    movies_df = pd.read_csv('movie.csv', dtype={'movieId': 'int32', 'title': 'str', 'genres': 'str'})
    ratings_df = pd.read_csv('rating.csv', dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'}, parse_dates=['timestamp'])
    tags_df = pd.read_csv('tag.csv', dtype={'userId': 'int32', 'movieId': 'int32', 'tag': 'str'}, parse_dates=['timestamp'])
    genome_tags_df = pd.read_csv('genome_tags.csv', dtype={'tagId': 'int32', 'tag': 'str'})
    genome_scores_df = pd.read_csv('genome_scores.csv', dtype={'movieId': 'int32', 'tagId': 'int32', 'relevance': 'float32'})
    
    return movies_df, ratings_df, tags_df, genome_tags_df, genome_scores_df

@memory.cache
def preprocess_data():
    movies_df, ratings_df, tags_df, genome_tags_df, genome_scores_df = load_data()
    movies_df['genres'] = movies_df['genres'].str.replace('|', ' ')
    
    # Pre-filter active users and popular movies
    active_users = ratings_df['userId'].value_counts().index[:10000]
    popular_movies = ratings_df['movieId'].value_counts().index[:1000]
    filtered_ratings = ratings_df[
        (ratings_df['userId'].isin(active_users)) & 
        (ratings_df['movieId'].isin(popular_movies))
    ]
    
    return movies_df, filtered_ratings, None

@memory.cache
def build_ratings_matrix(ratings_df):
    user_movie_matrix = ratings_df.pivot_table(
        index='userId', 
        columns='movieId', 
        values='rating', 
        fill_value=0
    )
    return csr_matrix(user_movie_matrix.values)

@memory.cache
def build_nearest_neighbors(matrix, n_neighbors=50, metric='cosine'):
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm='brute', n_jobs=-1)
    nn.fit(matrix)
    return nn

@lru_cache(maxsize=1000)
def get_content_based_recommendations(movie_id_tuple, num_recommendations=5):
    movies_df = global_data.movies_df
    tfidf_matrix = global_data.tfidf_matrix
    content_nn = global_data.content_nn
    movie_indices = movies_df.set_index('movieId').index
    
    movie_ids = list(movie_id_tuple)
    recommended_movies = []
    
    for movie_id in movie_ids:
        if movie_id in movie_indices:
            movie_idx = movies_df[movies_df['movieId'] == movie_id].index[0]
            distances, indices = content_nn.kneighbors(
                tfidf_matrix[movie_idx:movie_idx+1], 
                n_neighbors=501
            )
            
            # Convert distances to similarities (1 - distance for cosine)
            similarities = 1 - distances[0]
            
            recommended_movies.extend({
                'movieId': int(movies_df.iloc[idx]['movieId']),
                'title': movies_df.iloc[idx]['title'],
                'score': float(sim) * 5,
                'reason': 'Content-Based Filtering'
            } for sim, idx in zip(similarities[1:], indices[0][1:]))
    
    return recommended_movies

def get_collaborative_recommendations(user_id, num_recommendations=5, current_movie=None):
    user_index = user_id - 1
    collaborative_nn = global_data.collaborative_nn
    
    # Find similar users
    distances, indices = collaborative_nn.kneighbors(
        global_data.ratings_matrix[user_index:user_index+1], 
        n_neighbors=50
    )
    
    similarities = 1 - distances[0]
    similar_users = indices[0]
    
    ratings_matrix = global_data.ratings_matrix
    movies_df = global_data.movies_df
    
    # Vectorized prediction calculation
    user_ratings = ratings_matrix[similar_users].toarray()
    similarity_matrix = similarities[:, np.newaxis]
    predicted_ratings = (user_ratings.T @ similarity_matrix).flatten()
    predicted_ratings /= np.sum(similarities)
    
    # Create recommendations using numpy operations
    nonzero_predictions = predicted_ratings > 0
    user_unrated = ratings_matrix[user_index].toarray().flatten() == 0
    valid_predictions = nonzero_predictions & user_unrated
    
    movie_indices = np.where(valid_predictions)[0]
    sorted_indices = movie_indices[np.argsort(-predicted_ratings[valid_predictions])]
    
    recommended_movies = [{
        'movieId': int(idx + 1),
        'title': movies_df.iloc[idx]['title'],
        'predictedRating': float(predicted_ratings[idx]),
        'reason': 'Collaborative Filtering'
    } for idx in sorted_indices[:501]]
    
    if current_movie is not None:
        recommended_movies = [rec for rec in recommended_movies if rec['movieId'] != current_movie]
    
    return recommended_movies

def get_hybrid_recommendations(user_id, weights, num_recommendations=10, current_movie=None):
    user_index = user_id - 1
    user_ratings = global_data.ratings_matrix[user_index].toarray().flatten()
    user_positive_ratings = tuple(global_data.movies_df.iloc[i]['movieId'] 
                                for i in np.where(user_ratings >= 4)[0])
    
    content_recs = get_content_based_recommendations(user_positive_ratings, num_recommendations)
    collaborative_recs = get_collaborative_recommendations(user_id, num_recommendations, current_movie)
    
    # Use numpy operations for faster processing
    content_dict = {rec['movieId']: rec for rec in content_recs}
    collab_dict = {rec['movieId']: rec for rec in collaborative_recs}
    
    # Find common movies using set intersection
    common_movies = set(content_dict.keys()) & set(collab_dict.keys())
    
    # Combine recommendations using vectorized operations
    combined_scores = np.zeros(len(common_movies))
    movie_ids = np.array(list(common_movies))
    
    content_scores = np.array([content_dict[mid]['score'] for mid in movie_ids])
    collab_scores = np.array([collab_dict[mid]['predictedRating'] for mid in movie_ids])
    
    combined_scores = (content_scores * weights['content'] + 
                      collab_scores * weights['collaborative'])
    
    # Sort and create final recommendations
    sorted_indices = np.argsort(-combined_scores)
    combined_recs = [{
        'movieId': int(movie_ids[idx]),
        'title': content_dict[movie_ids[idx]]['title'],
        'score': float(combined_scores[idx]),
        'reason': 'Combined Content and Collaborative Filtering'
    } for idx in sorted_indices[:num_recommendations]]
    
    return (
        content_recs[:100],
        collaborative_recs[:100],
        combined_recs
    )

@app.before_request
def initialize_data():
    # Initialize all global data
    global_data.movies_df, ratings_df, _ = preprocess_data()
    global_data.ratings_matrix = build_ratings_matrix(ratings_df)
    
    # Build TF-IDF matrix for content-based filtering
    tfidf = TfidfVectorizer(stop_words='english')
    global_data.tfidf_matrix = tfidf.fit_transform(global_data.movies_df['genres'])
    
    # Build nearest neighbors models
    global_data.content_nn = build_nearest_neighbors(global_data.tfidf_matrix)
    global_data.collaborative_nn = build_nearest_neighbors(global_data.ratings_matrix)

@app.route('/recommendations', methods=['POST'])
def recommendations():
    start_time = time.time()
    try:
        data = request.json
        user_id = data.get('userIndex')
        current_movie = data.get('currentMovie')
        weights = data.get('weights', {'content': 0.5, 'collaborative': 0.5})
        num_recommendations = data.get('numRecommendations', 10)

        if user_id is None or not isinstance(user_id, int):
            return jsonify({'error': 'Invalid user index'}), 400

        if user_id > global_data.ratings_matrix.shape[0]:
            return jsonify({'error': 'Invalid user index'}), 400

        content_recommendations, collaborative_recommendations, hybrid_recommendations = get_hybrid_recommendations(
            user_id,
            weights,
            num_recommendations,
            current_movie
        )
        
        end_time = time.time()
        
        return jsonify({
            'content_recommendations': content_recommendations,
            'collaborative_recommendations': collaborative_recommendations,
            'recommendations': hybrid_recommendations,
            'response_time': end_time - start_time,
            'currentMovie': current_movie
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=3000, debug=True)