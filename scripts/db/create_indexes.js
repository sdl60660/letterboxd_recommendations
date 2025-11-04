// Movies
db.movies.createIndex({ movie_id: 1 }, { unique: true });

// Ratings
db.ratings.createIndex({ user_id: 1, movie_id: 1 }, { unique: true });
db.ratings.createIndex({ movie_id: 1 });
db.ratings.createIndex({ user_id: 1 });

// Users
db.users.createIndex({ username: 1 }, { unique: true });

// Redirect helper
db.movie_redirects.createIndex({ old_id: 1 });
