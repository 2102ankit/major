const express = require("express");
const fs = require("fs");
const path = require("path");
const csv = require("csv-parser");
const ContentBasedRecommender = require("content-based-recommender");

const app = express();
app.use(express.json());

// Initialize the content-based recommender
let movieData = []; // To store movieId, title, and genres from movies.csv
let genomeData = {}; // To store genome tags relevance by movieId from genome_scores.csv

const recommender = new ContentBasedRecommender({
  minScore: 0.1, // Minimum similarity score to be considered
  maxSimilarDocuments: 5, // Max similar items to retrieve
});

// Function to read and process movie data for content-based filtering
function loadMoviesAndGenomeData(callback) {
  fs.createReadStream(path.join(__dirname, "movie.csv"))
    .pipe(csv())
    .on("data", (row) => {
      movieData.push({
        id: row.movieId,
        content: row.genres,
      });
    })
    .on("end", () => {
      recommender.train(movieData);
      loadGenomeData(callback);
    });
}

// Function to load genome scores for movies
function loadGenomeData(callback) {
  fs.createReadStream(path.join(__dirname, "genome_scores.csv"))
    .pipe(csv())
    .on("data", (row) => {
      if (!genomeData[row.movieId]) genomeData[row.movieId] = [];
      genomeData[row.movieId].push({
        tagId: row.tagId,
        relevance: row.relevance,
      });
    })
    .on("end", callback);
}

// Function to get content-based recommendations
// function getContentBasedRecommendations(itemIds) {

//   let contentRecommendations = [];
//   itemIds.forEach((itemId) => {
//     const similarItems = recommender.getSimilarDocuments(itemId, 0, 5);
//     similarItems.forEach((similarItem) => {
//       contentRecommendations.push({
//         item: similarItem.id,
//         score: similarItem.score,
//         reason: "Content-Based Filtering",
//       });
//     });
//   });

//   const uniqueRecommendations = contentRecommendations.reduce(
//     (acc, current) => {
//       const exists = acc.find((item) => item.item === current.item);
//       if (!exists) acc.push(current);
//       return acc;
//     },
//     []
//   );

//   return uniqueRecommendations;
// }

function getContentBasedRecommendations(itemIds) {
  const documents = itemIds.map((id) => {
    const movie = movieData.find((movie) => movie.id === id);
    return { id: movie.id, content: movie.content };
  });

  const tempRecommender = new ContentBasedRecommender();
  tempRecommender.train(documents);

  const recommendations = tempRecommender.getSimilarDocuments(itemId, 0, 5);
  // return recommendations
}

// Function to calculate cosine similarity between two users
function calculateSimilarity(user1, user2) {
  let sumUser1 = 0,
    sumUser2 = 0,
    dotProduct = 0;

  for (let i = 0; i < user1.length; i++) {
    dotProduct += user1[i] * user2[i];
    sumUser1 += user1[i] * user1[i];
    sumUser2 += user2[i] * user2[i];
  }

  const magnitudeUser1 = Math.sqrt(sumUser1);
  const magnitudeUser2 = Math.sqrt(sumUser2);

  if (magnitudeUser1 === 0 || magnitudeUser2 === 0) return 0;

  return dotProduct / (magnitudeUser1 * magnitudeUser2);
}

// Function to get collaborative filtering recommendations
function getCollaborativeRecommendations(userIndex, matrix) {
  const targetUser = matrix[userIndex];
  let recommendations = Array(targetUser.length).fill(0);
  let similarityScores = [];

  for (let i = 0; i < matrix.length; i++) {
    if (i !== userIndex) {
      const similarity = calculateSimilarity(targetUser, matrix[i]);
      similarityScores.push({ user: i, similarity });
    }
  }

  similarityScores.sort((a, b) => b.similarity - a.similarity);

  for (let itemIndex = 0; itemIndex < targetUser.length; itemIndex++) {
    if (targetUser[itemIndex] === 0) {
      let weightedSum = 0,
        similaritySum = 0;
      for (const { user, similarity } of similarityScores) {
        if (matrix[user][itemIndex] !== 0) {
          weightedSum += matrix[user][itemIndex] * similarity;
          similaritySum += similarity;
        }
      }
      if (similaritySum !== 0) {
        recommendations[itemIndex] = weightedSum / similaritySum;
      }
    }
  }

  return recommendations
    .map((predictedRating, index) => ({
      item: index.toString(),
      predictedRating,
      reason: "Collaborative Filtering",
    }))
    .filter((rec) => rec.predictedRating > 0);
}

// Hybrid recommender system
function getHybridRecommendations(userIndex, matrix, userPositiveRatings) {
  const collaborativeRecs = getCollaborativeRecommendations(userIndex, matrix);
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

// Function to build the user-item rating matrix from CSV
function buildRatingsMatrix(filePath, callback) {
  const matrix = [];
  const userMap = new Map();

  fs.createReadStream(filePath)
    .pipe(csv())
    .on("data", (row) => {
      const userId = parseInt(row.userId) - 1;
      const movieId = parseInt(row.movieId) - 1;
      const rating = parseFloat(row.rating);

      if (!userMap.has(userId)) {
        userMap.set(userId, Array(10000).fill(0)); // Assuming 10k movies max
      }
      userMap.get(userId)[movieId] = rating;
    })
    .on("end", () => {
      userMap.forEach((ratingsArray, userId) => {
        matrix[userId] = ratingsArray;
      });
      callback(matrix);
    });
}

// API to get hybrid recommendations for a user
app.post("/recommendations", (req, res) => {
  const { userIndex } = req.body;

  if (!userIndex || userIndex < 0) {
    return res.status(400).json({ error: "Invalid user index" });
  }

  buildRatingsMatrix(path.join(__dirname, "ratings.csv"), (ratingsMatrix) => {
    if (userIndex >= ratingsMatrix.length) {
      return res.status(400).json({ error: "Invalid user index" });
    }

    const userPositiveRatings = [];
    ratingsMatrix[userIndex].forEach((rating, index) => {
      if (rating >= 4) {
        userPositiveRatings.push(index.toString());
      }
    });

    const hybridRecommendations = getHybridRecommendations(
      userIndex,
      ratingsMatrix,
      userPositiveRatings
    );

    res.json(hybridRecommendations);
  });
});

// Start the server
const PORT = 3000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});

// Load movie and genome data on server start
loadMoviesAndGenomeData(() => {
  console.log("Movies and genome data loaded, recommender system ready.");
});
