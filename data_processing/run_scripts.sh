
echo "Scraping users from site and adding to database"
python get_users.py
echo "Finished scraping users from site"
echo

echo "Scraping user ratings from site. Even running asynchronously, this can take several hours."
python get_ratings.py
echo "Finished scraping user ratings from site"
echo

echo "Scraping movie data from site."
python get_movies.py
echo "Finished scraping movie data from site"
echo

echo "================================"
echo "Creating Training Data Sample..."
echo "================================"
python create_training_data.py

echo "================="
echo "Building Model..."
echo "================="
python build_model.py

echo "================"
echo "Running Model..."
echo "================"
python run_model.py

echo "========="
echo "Finished!"
echo "========="