import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from scipy.sparse import csr_matrix, vstack
import time
import hashlib
import pickle
import os
from joblib import Memory
from functools import lru_cache
import threading
import logging
from sklearn.neighbors import NearestNeighbors

# Initialize memory caching
memory = Memory(location='.cache', verbose=0)
app = Flask(__name__)
app.config['DEBUG'] = True
app.logger.setLevel(logging.DEBUG)
CORS(app)  # Enable CORS for all routes

CACHE_DIR = '.recommendation_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

# Global variables for pre-computed data
global_data = threading.local()

def generate_cache_key(user_id, current_movie, weights):
    """Generate a unique cache key for recommendations"""
    key_components = [
        str(user_id), 
        str(current_movie), 
        str(weights.get('content', 0.5)), 
        str(weights.get('collaborative', 0.5))
    ]
    return hashlib.md5('_'.join(key_components).encode()).hexdigest()

def save_recommendations_cache(key, recommendations):
    """Save recommendations to persistent cache"""
    cache_path = os.path.join(CACHE_DIR, f'{key}_recommendations.pkl')
    with open(cache_path, 'wb') as f:
        pickle.dump(recommendations, f)

def load_recommendations_cache(key):
    """Load recommendations from persistent cache"""
    cache_path = os.path.join(CACHE_DIR, f'{key}_recommendations.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return None

@memory.cache
def load_link_data():
    return pd.read_csv('link.csv', dtype={'movieId': 'int32', 'imdbId': 'int32', 'tmdbId': 'float64'})

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

# @lru_cache(maxsize=1000)
def get_content_based_recommendations(movie_id_tuple, num_recommendations=5, current_movie=None):
    movies_df = global_data.movies_df
    tfidf_matrix = global_data.tfidf_matrix
    content_nn = global_data.content_nn
    
    recommended_movies = []
    
    # First, get recommendations based on current movie if provided
    if current_movie is not None:
        # Find the correct index for current movie using movieId
        try:
            current_movie_idx = movies_df[movies_df['movieId'] == current_movie].index[0]
            current_movie_title = movies_df.iloc[current_movie_idx]['title']
            
            distances, indices = content_nn.kneighbors(
                tfidf_matrix[current_movie_idx:current_movie_idx+1],
                n_neighbors=501
            )
            
            similarities = 1 - distances[0]
            
            # Add recommendations from current movie with high weight
            current_movie_recs = [{
                'movieId': int(movies_df.iloc[idx]['movieId']),
                'title': movies_df.iloc[idx]['title'],
                'score': float(sim)*5,
                'reason': f'Similar to current movie: {current_movie_title}'
            } for sim, idx in zip(similarities[1:], indices[0][1:])
            if int(movies_df.iloc[idx]['movieId']) != current_movie]
            
            recommended_movies.extend(current_movie_recs)
        except IndexError:
            print(f"Movie ID {current_movie} not found in dataset")
    
    # Then get recommendations based on user's highly rated movies
    for movie_id in movie_id_tuple:
        try:
            # Find the correct index for each movie using movieId
            movie_idx = movies_df[movies_df['movieId'] == movie_id].index[0]
            movie_title = movies_df.iloc[movie_idx]['title']
            
            distances, indices = content_nn.kneighbors(
                tfidf_matrix[movie_idx:movie_idx+1],
                n_neighbors=101
            )
            
            similarities = 1 - distances[0]
            
            user_pref_recs = [{
                'movieId': int(movies_df.iloc[idx]['movieId']),
                'title': movies_df.iloc[idx]['title'],
                'score': float(sim) * 0.5,
                'reason': f'Based on your interest in: {movie_title}'
            } for sim, idx in zip(similarities[1:], indices[0][1:])
            if int(movies_df.iloc[idx]['movieId']) != current_movie]
            
            recommended_movies.extend(user_pref_recs)
        except IndexError:
            print(f"Movie ID {movie_id} not found in dataset")
    
    # Sort by score and remove duplicates
    seen_movies = set()
    unique_recommendations = []
    for rec in sorted(recommended_movies, key=lambda x: x['score'], reverse=True):
        if rec['movieId'] not in seen_movies:
            seen_movies.add(rec['movieId'])
            unique_recommendations.append(rec)
    
    return unique_recommendations
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

    cache_key = generate_cache_key(user_id, current_movie, weights)
    cached_recommendations = load_recommendations_cache(cache_key)
    if cached_recommendations:
        return cached_recommendations
    
    user_index = user_id - 1
    user_ratings = global_data.ratings_matrix[user_index].toarray().flatten()
    user_positive_ratings = tuple(global_data.movies_df.iloc[i]['movieId'] 
                                for i in np.where(user_ratings >= 4)[0])
    
    content_recs = get_content_based_recommendations(user_positive_ratings, num_recommendations, current_movie)
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
        'reason': f'Similar to current movie and matches your preferences',
        'tmdbId': int(global_data.link_df[global_data.link_df['movieId'] == movie_ids[idx]]['tmdbId'].values[0])
    } for idx in sorted_indices[:num_recommendations]]
    
    # Add novelty recommendations (10% of total recommendations)
    novelty_count = max(1, num_recommendations // 5)
    
    # Select random movies not in existing recommendations
    existing_movie_ids = set(rec['movieId'] for rec in combined_recs)
    all_movie_ids = set(global_data.movies_df['movieId'])
    user_rated_movies = set(global_data.movies_df.iloc[np.where(user_ratings >= 4)[0]]['movieId'])
    novel_movie_candidates = list(all_movie_ids - existing_movie_ids - user_rated_movies)
    np.random.shuffle(novel_movie_candidates)
    
    novelty_recs = [{
        'movieId': int(movie_id),
        'title': global_data.movies_df[global_data.movies_df['movieId'] == movie_id]['title'].values[0],
        'score': -1,  # Explicitly set to -1 to indicate novelty
        'reason': 'Novelty Recommendation',
        'tmdbId': int(global_data.link_df[global_data.link_df['movieId'] == movie_id]['tmdbId'].values[0])
    } for movie_id in novel_movie_candidates[:novelty_count]]
    
    # Combine and return recommendations
    combined_recs.extend(novelty_recs)
    
    # return (
    #     content_recs[:100],
    #     collaborative_recs[:100],
    #     combined_recs
    # )
    # Save to cache before returning
    recommendations = (
        content_recs[:100],
        collaborative_recs[:100],
        combined_recs
    )
    save_recommendations_cache(cache_key, recommendations)
    
    return recommendations
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

    global_data.link_df = load_link_data()
    global_data.link_df['tmdbId'] = global_data.link_df['tmdbId'].fillna(0).astype('int32')


@app.route('/movies', methods=['GET'])
def get_movies():
    movies_df = global_data.movies_df
    movies_list = movies_df[['movieId', 'title']].to_dict(orient='records')
    return jsonify(movies_list)


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
        # return jsonify({
        #         'content_recommendations': content_recommendations,
        #         'collaborative_recommendations': collaborative_recommendations,
        #         'recommendations': hybrid_recommendations,
        #         'response_time': end_time - start_time,
        #         'currentMovie': current_movie
        #     })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=3000, debug=True)