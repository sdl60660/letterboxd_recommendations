# Letterboxd Recommendations

This project scrapes publicly-accessible Letterboxd data and creates a movie recommendation model with it that can generate recommendations when provided with a Letterboxd username.

Live project lives here: https://bit.ly/letterboxd-movie-recs

To build this I used:
* Python (requests/BeautifulSoup/asyncio/aiohttp) to scrape review data
* MongoDB (pymongo) to store user/rating/movie data
* Flask as a web server
* HTML/CSS/Javascript on the front-end
* Redis/redis queue for managing queued tasks (scraping user data, building/running the model)
* Heroku for hosting
