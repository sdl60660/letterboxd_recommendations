import os

if os.getcwd().endswith("data_processing"):
    from db_connect import connect_to_db

else:
    from data_processing.db_connect import connect_to_db


def main():
    # Connect to MongoDB client
    db_name, client = connect_to_db()

    # Find letterboxd database and user collection
    db = client[db_name]
    users = db.users


if __name__ == "__main__":
    main()
