git pull origin main

pipenv run python get_users.py
pipenv run python get_ratings.py
pipenv run python get_movies.py

pipenv run python create_training_data.py

git add .
git commit -m "Automated data update from EC2"
git push origin main