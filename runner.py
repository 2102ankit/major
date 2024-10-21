import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request, jsonify
from scipy.sparse import csr_matrix

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
    sparse_user_movie_matrix = csr_matrix(user_movie_matrix.values)
    return sparse_user_movie_matrix

# Content-Based Recommendation using Genres
def get_content_based_recommendations(movie_ids, movies_df, num_recommendations=5):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    movie_tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['genres'])
    genre_similarity = cosine_similarity(movie_tfidf_matrix)
    
    recommended_movies = []
    for movie_id in movie_ids:
        movie_idx = movies_df[movies_df['movieId'] == movie_id].index[0]
        similar_movies = list(enumerate(genre_similarity[movie_idx]))
        similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:num_recommendations + 1]
        
        for movie in similar_movies:
            recommended_movies.append({
                'movieId': int(movies_df.iloc[movie[0]]['movieId']),
                'title': movies_df.iloc[movie[0]]['title'],
                'score': float(movie[1]),
                'reason': 'Content-Based Filtering'
            })
    
    return recommended_movies

# Collaborative Filtering using Cosine Similarity
def get_collaborative_recommendations(user_id, ratings_matrix, movies_df, num_recommendations=5):
    user_similarity = cosine_similarity(ratings_matrix)
    user_index = user_id - 1
    user_ratings = ratings_matrix[user_index].toarray().flatten()
    similar_users = list(enumerate(user_similarity[user_index]))
    similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)[1:]
    
    recommended_movies = []
    
    for user, sim_score in similar_users:
        user_ratings = ratings_matrix[user].toarray().flatten()
        unseen_movies = np.where(user_ratings == 0)[0]
        
        for movie_id in unseen_movies:
            predicted_rating = sim_score * user_ratings[movie_id]
            if predicted_rating > 0:
                recommended_movies.append({
                    'movieId': int(movie_id + 1),
                    'title': movies_df.iloc[movie_id]['title'],
                    'predictedRating': float(predicted_rating),
                    'reason': 'Collaborative Filtering'
                })
    
    return sorted(recommended_movies, key=lambda x: x['predictedRating'], reverse=True)[:num_recommendations]

# Hybrid Recommendation System
def get_hybrid_recommendations(user_id, ratings_matrix, movies_df, weights, num_recommendations=10):
    user_index = user_id - 1
    user_ratings = ratings_matrix[user_index].toarray().flatten()
    user_positive_ratings = np.where(user_ratings >= 4)[0].tolist()
    
    content_recs = get_content_based_recommendations([movies_df.iloc[i]['movieId'] for i in user_positive_ratings], movies_df, num_recommendations)
    collaborative_recs = get_collaborative_recommendations(user_id, ratings_matrix, movies_df, num_recommendations)
    
    # Combine recommendations based on weights
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
    
    return sorted(combined_recs.values(), key=lambda x: x['score'], reverse=True)[:num_recommendations]

# API to get hybrid recommendations
@app.route('/recommendations', methods=['POST'])
def recommendations():
    try:
        data = request.json
        user_id = data.get('userIndex')
        weights = data.get('weights', {'content': 0.5, 'collaborative': 0.5})  # Default weights
        num_recommendations = data.get('numRecommendations', 10)  # Default number of recommendations

        if user_id is None or not isinstance(user_id, int):
            return jsonify({'error': 'Invalid user index'}), 400
        
        movies_df, ratings_df, _ = preprocess_data()
        ratings_matrix = build_ratings_matrix(ratings_df)
        
        if user_id > ratings_matrix.shape[0]:
            return jsonify({'error': 'Invalid user index'}), 400
        
        hybrid_recommendations = get_hybrid_recommendations(user_id, ratings_matrix, movies_df, weights, num_recommendations)
        
        return jsonify(hybrid_recommendations)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Start the Flask server
if __name__ == '__main__':
    app.run(port=3000, debug=True)
