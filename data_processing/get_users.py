import requests
from bs4 import BeautifulSoup

import pymongo
from config import config

db_name = config["MONGO_DB"]
client = pymongo.MongoClient(f'mongodb+srv://{config["MONGO_USERNAME"]}:{config["MONGO_PASSWORD"]}@cluster0.{config["MONGO_CLUSTER_ID"]}.mongodb.net/{db_name}?retryWrites=true&w=majority')

db = client[db_name]
users = db.users

base_url = "https://letterboxd.com/members/popular/page/{}"
datafile = open('data/users.txt', 'w')

for page in range(1, 1501):
    print("Page {}".format(page))

    r = requests.get(base_url.format(page))
    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table", attrs={"class": "person-table"})
    rows = table.findAll("td", attrs={"class": "table-person"})
    
    for row in rows:
        link = row.find("a")["href"]
        username = link.strip('/')
        display_name = row.find("a", attrs={"class": "name"}).text.strip()
        num_reviews = int(row.find("small").find("a").text.replace('\xa0', ' ').split()[0].replace(',', ''))

        user = {
            "username": username,
            "display_name": display_name,
            "num_reviews": num_reviews
        }

        datafile.write(username + '\n')
        users.update_one({"username": user["username"]}, {"$set": user}, upsert=True)
        
datafile.close()