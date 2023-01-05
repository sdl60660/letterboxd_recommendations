def connect_to_db():
    import os
    import pymongo
    
    try:
        # This is a MongoDB connection URL and a TMDB API key imported from a file in the .gitignore
        if os.getcwd().endswith("data_processing"):
            from db_config import config, tmdb_key
        else:
            from data_processing.db_config import config, tmdb_key

        db_name = config["MONGO_DB"]
        if "CONNECTION_URL" in config.keys():
            client = pymongo.MongoClient(config["CONNECTION_URL"], server_api=pymongo.server_api.ServerApi('1'))
        else:
            client = pymongo.MongoClient(f'mongodb+srv://{config["MONGO_USERNAME"]}:{config["MONGO_PASSWORD"]}@cluster0.{config["MONGO_CLUSTER_ID"]}.mongodb.net/{db_name}?retryWrites=true&w=majority')

    except ModuleNotFoundError:
        # If not running locally, since db_config data is not committed to git
        import os
        db_name = os.environ['MONGO_DB']
        client = pymongo.MongoClient(os.environ["CONNECTION_URL"], server_api=pymongo.server_api.ServerApi('1'))
        tmdb_key = os.environ['TMDB_KEY']
    
    return db_name, client, tmdb_key