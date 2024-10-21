const express = require("express");
const fs = require("fs");
const path = require("path");
const csv = require("csv-parser");
const ContentBasedRecommender = require("content-based-recommender");

const app = express();
app.use(express.json());

const PORT = 3000;

const recommender = new ContentBasedRecommender({
  minScore: 0.1, // Minimum similarity score to be considered
  maxSimilarDocuments: 5, // Max similar items to retrieve
});

// File paths
const MOVIES_CSV_PATH = path.join(__dirname, "movie.csv");
const RATINGS_CSV_PATH = path.join(__dirname, "rating.csv");

// In-memory cache for essential data
let movieData = [];
let ratingsMatrix = {};
let maxMovieId = 0;
let maxUserId = 0;

// Streaming and lazy loading for movies data
function loadMoviesDataInChunks(callback) {
  if (movieData.length > 0) {
    callback();
    return;
  }

  console.log("Streaming movies data...");
  const movieStream = fs.createReadStream(MOVIES_CSV_PATH).pipe(csv());

  movieStream.on("data", (row) => {
    // Process each row and build movie data chunk by chunk
    movieData.push({
      id: row.movieId,
      content: row.genres.replace(/\|/g, " "), // Use genres for content-based filtering
    });

    if (movieData.length >= 1000) {
      // If movie data exceeds 1000, process it and reset for next chunk
      trainRecommenderWithMovies(movieData);
      movieData = [];
    }
  });

  movieStream.on("end", () => {
    if (movieData.length > 0) {
      // Process remaining movie data if any
      trainRecommenderWithMovies(movieData);
    }
    console.log("Movies data streaming complete.");
    callback();
  });
}

// Train the content-based recommender with movie chunks
function trainRecommenderWithMovies(movieChunk) {
  const recommender = new ContentBasedRecommender({
    minScore: 0.1,
    maxSimilarDocuments: 5,
  });
  recommender.train(movieChunk);
}

// Get content-based recommendations in chunks
function getContentBasedRecommendations(itemIds) {
  const recommendations = [];
  itemIds.forEach((itemId) => {
    const similarItems = recommender.getSimilarDocuments(itemId, 0, 5);
    similarItems.forEach((similarItem) => {
      recommendations.push({
        item: similarItem.id,
        score: similarItem.score,
        reason: "Content-Based Filtering",
      });
    });
  });
  return recommendations.filter(
    (rec, index, self) => index === self.findIndex((r) => r.item === rec.item)
  );
}

// Streaming and lazy loading for ratings data
function loadRatingsMatrixInChunks(userIndex, callback) {
  const ratingsStream = fs.createReadStream(RATINGS_CSV_PATH).pipe(csv());

  console.log("Streaming ratings data...");
  ratingsStream.on("data", (row) => {
    const userId = parseInt(row.userId) - 1;
    const movieId = parseInt(row.movieId) - 1;
    const rating = parseFloat(row.rating);

    if (!ratingsMatrix[userId]) ratingsMatrix[userId] = {};
    ratingsMatrix[userId][movieId] = rating;

    // Keep track of max user and movie IDs
    maxUserId = Math.max(maxUserId, userId);
    maxMovieId = Math.max(maxMovieId, movieId);

    if (Object.keys(ratingsMatrix).length >= 1000) {
      // Process chunk of ratings and free memory
      processRatingsChunk(userIndex, ratingsMatrix, callback);
      ratingsMatrix = {};
    }
  });

  ratingsStream.on("end", () => {
    if (Object.keys(ratingsMatrix).length > 0) {
      processRatingsChunk(userIndex, ratingsMatrix, callback);
    }
    console.log("Ratings data streaming complete.");
    callback();
  });
}

// Process ratings chunk and free memory after use
function processRatingsChunk(userIndex, matrix, callback) {
  const userRatings = matrix[userIndex] || {};

  // Convert ratings to a matrix format for similarity calculation
  const userPositiveRatings = [];
  Object.entries(userRatings).forEach(([movieId, rating]) => {
    if (rating >= 4) {
      userPositiveRatings.push(movieId);
    }
  });

  // Continue with recommendation process
  callback(userPositiveRatings);
}

// Hybrid recommender system (combine collaborative and content-based recommendations)
function getHybridRecommendations(userIndex, userPositiveRatings) {
  const collaborativeRecs = getCollaborativeRecommendations(userIndex);
  const contentBasedRecs = getContentBasedRecommendations(userPositiveRatings);

  let combinedRecs = [...collaborativeRecs];

  contentBasedRecs.forEach((contentRec) => {
    const existsInCollaborative = combinedRecs.find(
      (rec) => rec.item === contentRec.item
    );
    if (!existsInCollaborative) {
      combinedRecs.push(contentRec);
    }
  });

  return combinedRecs.sort(
    (a, b) => (b.predictedRating || b.score) - (a.predictedRating || a.score)
  );
}

// Collaborative filtering recommendations
function getCollaborativeRecommendations(userIndex) {
  // Collaborative filtering using ratings data chunk
  const userRatings = ratingsMatrix[userIndex] || {};
  let recommendations = [];

  Object.entries(userRatings).forEach(([movieId, rating]) => {
    if (rating === 0) {
      recommendations.push({
        item: movieId,
        predictedRating: Math.random() * 5, // Dummy collaborative score
        reason: "Collaborative Filtering",
      });
    }
  });

  return recommendations;
}

// API to get hybrid recommendations for a user
app.post("/recommendations", (req, res) => {
  const { userIndex } = req.body;

  if (userIndex == null || userIndex < 0) {
    return res.status(400).json({ error: "Invalid user index" });
  }

  // Ensure only a single response is sent after all processing
  let sentResponse = false;

  // Load ratings matrix in chunks and respond with hybrid recommendations
  loadRatingsMatrixInChunks(userIndex, (userPositiveRatings) => {
    if (sentResponse) return; // Prevent multiple responses
    loadMoviesDataInChunks(() => {
      if (sentResponse) return; // Prevent multiple responses

      const hybridRecommendations = getHybridRecommendations(
        userIndex,
        userPositiveRatings
      );

      // Send the final response only once
      if (!sentResponse) {
        res.json(hybridRecommendations);
        sentResponse = true;
      }
    });
  });
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
