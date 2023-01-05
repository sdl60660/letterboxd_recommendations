# Letterboxd Recommendations

This project scrapes publicly-accessible Letterboxd data and creates a movie recommendation model with it that can generate recommendations when provided with a Letterboxd username.

Live project lives here: https://letterboxd.samlearner.com/

### Methodology

A user's "star" ratings are scraped from their Letterboxd profile and assigned numerical ratings from 1 to 10 (accounting for half stars). Their ratings are then combined with a sample of ratings from the top 4000 most active users on the site to create a collaborative filtering recommender model using singular value decomposition (SVD). All movies in the full dataset that the user has not rated are run through the model for predicted scores and the items with the top predicted scores are returned. Due to constraints in time and computing power, the maxiumum sample size that a user is allowed to select is 500,000 samples, though there are over five million ratings in the full dataset from the top 4000 Letterboxd users alone.

### Notes

The underlying model is completely blind to genres, themes, directors, cast, or any other content information; it recommends only based on similarities in rating patterns between users and movies. I've found that it tends to recommend very popular movies often, regardless of an individual user's taste ("basically everyone who watches 12 Angry Men seems to like it, so why wouldn't you?"). To help counteract that, I included a popularity filter that filters by how many times a movie has been rated in the dataset, so that users can specifically find more obscure recommendations. I've also found that it occasionally just completely whiffs (I guess most people who watch "Taylor Swift: Reputation Stadium Tour" do like it, but it's not really my thing). I think that's just the nature of the beast, to some extent, particularly when working with a relatively small sample. It'll return 50 recommendations and that's usually enough to work with if I'm looking for something to watch, even if there are a couple misses here or there.

### Running this on your own

The web crawling/data processing portion of this project (everything that isn't related to what happens on the webpage) lives in the `data_processing` subdirectory. There you'll find a bash script called `run_scripts.sh`. Use this as your guide for running the crawler, building a training data set, or running the model on your own. **However**, keep in mind that a full crawl of users, ratings, and movies will take several hours. If you'd like to skip that step, I'll keep [this Kaggle dataset](https://www.kaggle.com/samlearner/letterboxd-movie-ratings-data) up to date with the data from my latest crawl. Regardless of whether you run the crawl on your own or download the exported data from Kaggle, there are three very quick things you'll need to do to get up and running outside of installing the dependencies in `Pipfile` using pipenv:

1. Start up a local MongoDB server (ideally at the default port 27017)
2. Add a file to the data_processing subdirectory called "db_config" with some basic information on your MongoDB server. If you're running a local server on the default port, all you'd need in that file is this: `config = { 'MONGO_DB': 'letterboxd', 'CONNECTION_URL': 'mongodb://localhost:27017/'}`

At that point, if you'd like to run the crawl on your own, you can just run the first three scripts listed in `data_processing/run_scripts.sh` (`get_users.py`, `get_ratings.py`, `get_movies.py`). If you download the data from Kaggle, you'll just need to import each CSV into your Mongo database as its own collection. The other three python scripts (`create_training_data.py`, `build_model.py`, `run_model.py`) will build and run the SVD model for you.

If you'd like to run the web server with the front-end locally, you'll need to run a local Redis instance, as well. You can then run `pipenv run python worker.py` to activate the Redis worker in the background and run start the web server by running `pipenv run uvicorn main:app --reload`. Navigate into the `frontend` directory and run `npm install` to install packages and then `npm start` to start the frontend React app.


### Built With
* Python (requests/BeautifulSoup/asyncio/aiohttp) to scrape review data
* MongoDB (pymongo) to store user/rating/movie data
* FastAPI as a python web server
* HTML/CSS/Javascript/React/MaterialUI on the front-end
* Redis/redis queue for managing queued tasks (scraping user data, building/running the model)
* Heroku/Vercel for hosting
