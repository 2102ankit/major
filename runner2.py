import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request, jsonify
from scipy.sparse import csr_matrix
import time

app = Flask(__name__)

# Load Data
def load_data():
    movies_df = pd.read_csv('movie.csv', dtype={'movieId': 'int32', 'title': 'str', 'genres': 'str'})
    ratings_df = pd.read_csv('rating.csv', dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'}, parse_dates=['timestamp'])
    tags_df = pd.read_csv('tag.csv', dtype={'userId': 'int32', 'movieId': 'int32', 'tag': 'str'}, parse_dates=['timestamp'])
    genome_tags_df = pd.read_csv('genome_tags.csv', dtype={'tagId': 'int32', 'tag': 'str'})
    genome_scores_df = pd.read_csv('genome_scores.csv', dtype={'movieId': 'int32', 'tagId': 'int32', 'relevance': 'float32'})
    
    return movies_df, ratings_df, tags_df, genome_tags_df, genome_scores_df

# Preprocess Data
def preprocess_data():
    movies_df, ratings_df, tags_df, genome_tags_df, genome_scores_df = load_data()
    movies_df['genres'] = movies_df['genres'].str.replace('|', ' ')
    merged_genome_scores = pd.merge(genome_scores_df, genome_tags_df, on='tagId', how='inner')
    return movies_df, ratings_df, merged_genome_scores

# Build user-item rating matrix
def build_ratings_matrix(ratings_df):
    active_users = ratings_df['userId'].value_counts().index[:10000]
    popular_movies = ratings_df['movieId'].value_counts().index[:1000]
    filtered_ratings = ratings_df[(ratings_df['userId'].isin(active_users)) & (ratings_df['movieId'].isin(popular_movies))]
    user_movie_matrix = filtered_ratings.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)
    return csr_matrix(user_movie_matrix.values)

# Content-Based Recommendation using Genres
def get_content_based_recommendations(movie_ids, movies_df, num_recommendations=5):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    movie_tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['genres'])
    genre_similarity = cosine_similarity(movie_tfidf_matrix)

    movie_indices = movies_df.set_index('movieId').index
    recommended_movies = []

    for movie_id in movie_ids:
        if movie_id in movie_indices:
            movie_idx = movies_df[movies_df['movieId'] == movie_id].index[0]
            similar_movies = np.argsort(-genre_similarity[movie_idx])[1:501]
            recommended_movies.extend({
                'movieId': int(movies_df.iloc[sim_movie]['movieId']),
                'title': movies_df.iloc[sim_movie]['title'],
                'score': float(genre_similarity[movie_idx][sim_movie]) * 5,
                'reason': 'Content-Based Filtering'
            } for sim_movie in similar_movies)

    return recommended_movies

# Collaborative Filtering using Cosine Similarity
def get_collaborative_recommendations(user_id, ratings_matrix, movies_df, num_recommendations=5, current_movie=None):
    user_similarity = cosine_similarity(ratings_matrix)
    user_index = user_id - 1
    similar_users = np.argsort(-user_similarity[user_index])[:50]  # Limit similar users

    recommended_movies = []

    for movie_id in range(ratings_matrix.shape[1]):
        if ratings_matrix[user_index, movie_id] == 0:
            predicted_rating = 0
            total_similarity = 0

            for user in similar_users:
                if ratings_matrix[user, movie_id] > 0:
                    predicted_rating += user_similarity[user_index, user] * ratings_matrix[user, movie_id]
                    total_similarity += user_similarity[user_index, user]

            if total_similarity > 0:  # Ensure this is a scalar condition
                predicted_rating /= total_similarity  # No .item() needed since total_similarity is already scalar

            if predicted_rating > 0:
                recommended_movies.append({
                    'movieId': int(movie_id + 1),
                    'title': movies_df.iloc[movie_id]['title'],
                    'predictedRating': float(predicted_rating),
                    'reason': 'Collaborative Filtering'
                })

    if current_movie is not None:
        recommended_movies = [rec for rec in recommended_movies if rec['movieId'] != current_movie]

    return sorted(recommended_movies, key=lambda x: x['predictedRating'], reverse=True)[:501]

# Hybrid Recommendation System
def get_hybrid_recommendations(user_id, ratings_matrix, movies_df, weights, num_recommendations=10, current_movie=None):
    user_index = user_id - 1
    user_ratings = ratings_matrix[user_index].toarray().flatten()
    user_positive_ratings = np.where(user_ratings >= 4)[0].tolist()

    content_recs = get_content_based_recommendations(
        [movies_df.iloc[i]['movieId'] for i in user_positive_ratings],
        movies_df,
        num_recommendations
    )

    collaborative_recs = get_collaborative_recommendations(user_id, ratings_matrix, movies_df, num_recommendations, current_movie)

    combined_recs = {}
    for rec in content_recs:
        combined_recs[rec['movieId']] = {
            'movieId': rec['movieId'],
            'title': rec['title'],
            'score': rec['score'] * weights['content'],
            'reason': rec['reason']
        }

    for rec in collaborative_recs:
        if rec['movieId'] in combined_recs:
            combined_recs[rec['movieId']]['score'] += rec['predictedRating'] * weights['collaborative']
        else:
            combined_recs[rec['movieId']] = {
                'movieId': rec['movieId'],
                'title': rec['title'],
                'score': rec['predictedRating'] * weights['collaborative'],
                'reason': rec['reason']
            }

    sorted_recommendations = sorted(combined_recs.values(), key=lambda x: x['score'], reverse=True)[:num_recommendations]

    return content_recs[:100], collaborative_recs[:100], sorted_recommendations

# API to get hybrid recommendations
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

        movies_df, ratings_df, _ = preprocess_data()
        ratings_matrix = build_ratings_matrix(ratings_df)

        if user_id > ratings_matrix.shape[0]:
            return jsonify({'error': 'Invalid user index'}), 400

        content_recommendations, collaborative_recommendations, hybrid_recommendations = get_hybrid_recommendations(
            user_id,
            ratings_matrix,
            movies_df,
            weights,
            num_recommendations,
            current_movie
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        return jsonify({
            'content_recommendations': content_recommendations,
            'collaborative_recommendations': collaborative_recommendations,
            'recommendations': hybrid_recommendations,
            'response_time': response_time,
            'currentMovie': current_movie
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Start the Flask server
if __name__ == '__main__':
    app.run(port=3000, debug=True)
