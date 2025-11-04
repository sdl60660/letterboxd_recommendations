# Letterboxd Recommendations

This project scrapes publicly-accessible Letterboxd data and creates a movie recommendation model with it that can generate recommendations when provided with a Letterboxd username.

> **Live project:** [https://letterboxd.samlearner.com](https://letterboxd.samlearner.com)

## Methodology

A user's "star" ratings are scraped their Letterboxd profile and assigned numerical ratings from 1 to 10 (accounting for half stars). Their ratings are then folded into a model that's been trained on a sample of ratings from other users on the site to create a collaborative filtering recommender model using singular value decomposition (SVD). All movies in the full dataset that the user has not rated are run through the model for predicted scores and the items with the top predicted scores are returned. Due to constraints in time and computing power, the maxiumum sample size that a user is allowed to select contains about five million ratings, though there are over 50 million ratings in the full dataset.

## Notes

The underlying model is completely blind to genres, themes, directors, cast, or any other content information; it recommends only based on similarities in rating patterns between users and movies. I've found that it tends to recommend very popular movies often, regardless of an individual user's taste ("basically everyone who watches 12 Angry Men seems to like it, so why wouldn't you?"). I've also found that it occasionally just completely whiffs (I guess most people who watch "Taylor Swift: Reputation Stadium Tour" do like it, but it's not really my thing). I think that's just the nature of the beast, to some extent, particularly when working with a relatively small sample. It'll return 50 recommendations for a set of filters and that's usually enough to work with if I'm looking for something to watch, even if there are a couple misses here or there.

## Getting started

To run the full stack locally (backend, crawlers, frontend), you’ll need Docker and Node installed.

1. **Create your environment file**
   ```bash
   cp .env.example .env
   ```
   - Update any values as needed.  
   - `SECRET_KEY` can be anything.  
   - `CONNECTION_URL` (or `MONGODB_URI`) should point to your MongoDB instance — it defaults to the local Docker Mongo container (`mongodb://mongo:27017`).  
   - If you have a TMDB API key, add it under `TMDB_KEY` (optional but recommended for richer movie data).

2. **Run the setup script**  
   This installs Python dependencies (via Pipenv), frontend dependencies (via npm), installs pre-commit hooks, builds Docker images, starts Mongo + Redis, creates indexes, seeds minimal data, and runs quick tests:
   ```bash
   make bootstrap
   # or
   bash scripts/dev/bootstrap.sh
   ```

3. **Start all services**  
   ```bash
   make up
   # or
   docker compose up --build
   ```
   This will start:
   - the FastAPI web service at [http://localhost:8000](http://localhost:8000)
   - the React frontend at [http://localhost:3000](http://localhost:3000)

4. **Seed the database (first-time setup only)**  
   To populate your database with live Letterboxd and TMDB data:
   ```bash
   make seed-full
   # or
   bash scripts/db/seed_full.sh
   ```
   Alternatively, can also skip the crawling and start from (somewhat old) archived data on [Kaggle](https://www.kaggle.com/samlearner/letterboxd-movie-ratings-data).

Once everything’s running, visit `http://localhost:3000`, enter a Letterboxd username, and you should get recommendations powered by your local database.

## Troubleshooting

- **Mongo connection error**  
  Make sure Docker is running and `CONNECTION_URL` points to `mongodb://mongo:27017` (or your remote Mongo server)

- **TMDB key errors**  
  Live rich-data crawls and some smoke tests require a valid `TMDB_KEY`.  
  If missing, those tests will skip automatically.

- **Pre-commit hooks not running**  
  Run `pipenv run pre-commit install` once after cloning.

- **Frontend not loading**  
  Ensure `npm install` completed inside `frontend/` (the bootstrap script does this automatically).

## Project Structure

```
letterboxd_recommendations/
├── data_processing/       # Crawlers, model building, and scripts
│   ├── models/            # Cached compressed models live here, as well as the eval results for tuning
│   └── data/              # Cached data used for building models or attaching rich movie data, mainly stored as .parquet files
├── frontend/              # React app
├── scripts/
│   ├── dev/               # Setup scripts (bootstrap, wait-for, etc.)
│   └── db/                # Index + seeding scripts
├── tests/                 # Unit, integration, and smoke tests
├── docker-compose.yml
├── Makefile
└── .env.example
```

## Built With

- Python (requests/BeautifulSoup/asyncio/aiohttp) to scrape review data
- MongoDB (pymongo) to store user/rating/movie data
- FastAPI as a python web server
- HTML/CSS/Javascript/React/MaterialUI on the front-end
- Redis/redis queue for managing queued tasks (scraping user data, building/running the model)
- Heroku/Vercel for hosting
