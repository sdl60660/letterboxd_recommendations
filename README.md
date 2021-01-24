# Letterboxd Recommendations

This project scrapes publicly-accessible Letterboxd data and creates a movie recommendation model with it that can generate recommendations when provided with a Letterboxd username.

Live project lives here: https://bit.ly/letterboxd-movie-recs

### Methodology

A user's "star" ratings are scraped their Letterboxd profile and assigned numerical ratings from 1 to 10 (accounting for half stars). Their ratings are then combined with a sample of ratings from the top 4000 most active users on the site to create a collaborative filtering recommender model using singular value decomposition (SVD). All movies in the full dataset that the user has not rated are run through the model for predicted scores and the items with the top predicted scores are returned. Due to constraints in time and computing power, the maxiumum sample size that a user is allowed to select is 500,000 samples, though there are over five million ratings in the full dataset from the top 4000 Letterboxd users alone.

### Notes

The underlying model is completely blind to genres, themes, directors, cast, or any other content information; it recommends only based on similarities in rating patterns between users and movies. I've found that it tends to recommend very popular movies often, regardless of an individual user's taste ("basically everyone who watches 12 Angry Men seems to like it, so why wouldn't you?"). To help counteract that, I included a popularity filter that filters by how many times a movie has been rated in the dataset, so that users can specifically find more obscure recommendations. I've also found that it occasionally just completely whiffs (I guess most people who watch "Taylor Swift: Reputation Stadium Tour" do like it, but it's not really my thing). I think that's just the nature of the beast, to some extent, particularly when working with a relatively small sample. It'll return 50 recommendations and that's usually enough to work with if I'm looking for something to watch, even if there are a couple misses here or there.


### Built With
* Python (requests/BeautifulSoup/asyncio/aiohttp) to scrape review data
* MongoDB (pymongo) to store user/rating/movie data
* Flask as a web server
* HTML/CSS/Javascript on the front-end
* Redis/redis queue for managing queued tasks (scraping user data, building/running the model)
* Heroku for hosting
