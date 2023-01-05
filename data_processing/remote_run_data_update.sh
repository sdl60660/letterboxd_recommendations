# Sync from Github
git pull origin main

# Run DB updates
pipenv run python get_users.py
pipenv run python get_ratings.py
pipenv run python get_movies.py

# Generate new training data file
pipenv run python create_training_data.py

# Push to Github
git add .
git commit -m "Automated data update from EC2"
git push origin main

# Shutdown remote EC2 instance when finished
sudo shutdown -h now