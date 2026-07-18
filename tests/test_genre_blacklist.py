import sys
from unittest.mock import MagicMock

# Mock external libs
sys.modules["pandas"] = MagicMock()
sys.modules["pymongo"] = MagicMock()
sys.modules["aiohttp"] = MagicMock()

# Mock internal dependencies to avoid import chains
sys.modules["data_processing.get_user_ratings"] = MagicMock()
sys.modules["data_processing.model"] = MagicMock()
sys.modules["data_processing.utils.config"] = MagicMock()
sys.modules["data_processing.utils.utils"] = MagicMock()

import unittest
# Clean imports for the module under test
from data_processing.run_model import run_model

class TestGenreBlacklist(unittest.TestCase):
    def setUp(self):
        # Mock Algo
        self.algo = MagicMock()
        self.algo.rating_min = 0.5
        self.algo.rating_max = 5.0
        # Mock update_algo to return itself
        self.algo.update_algo.return_value = self.algo

        # Mock Data
        self.username = "testuser"
        self.user_data = [{"movie_id": "m1", "rating_val": 4.0}]
        self.sample_movie_list = ["m2", "m3", "m4"]
        
        # m2: Action (Keep)
        # m3: Horror (Exclude)
        # m4: Comedy (Keep)
        self.movie_data = {
            "m2": {"movie_id": "m2", "movie_title": "Action Movie", "genres": ["Action", "Adventure"]},
            "m3": {"movie_id": "m3", "movie_title": "Horror Movie", "genres": ["Horror", "Thriller"]},
            "m4": {"movie_id": "m4", "movie_title": "Comedy Movie", "genres": ["Comedy"]}
        }

    def test_run_model_filters_blacklist(self):
        # Setup mock predictions
        # (uid, iid, true_r, est, details)
        # Give Horror movie the highest score to ensure it would be recommended if not filtered
        predictions = [
            (self.username, "m3", 0, 5.0, {}),  # Horror (Highest score)
            (self.username, "m2", 0, 4.5, {}),  # Action
            (self.username, "m4", 0, 4.0, {}),  # Comedy
        ]
        self.algo.test.return_value = predictions

        # Run model with blacklist
        blacklist = "Horror, Documentary"
        results = run_model(
            self.username,
            self.algo,
            self.user_data,
            self.sample_movie_list,
            movie_data=self.movie_data,
            num_recommendations=5,
            fold_in=True,
            genre_blacklist=blacklist
        )

        # Verify Results
        # Should contain m2 and m4, but NOT m3
        movie_ids = [r["movie_id"] for r in results]
        
        print(f"Results: {movie_ids}")
        
        self.assertIn("m2", movie_ids)
        self.assertIn("m4", movie_ids)
        self.assertNotIn("m3", movie_ids)
        
        # Verify m3 was excluded despite high score
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["movie_id"], "m2") # Highest remaining score

    def test_run_model_no_blacklist(self):
        # Run without blacklist to confirm baseline
        predictions = [
            (self.username, "m3", 0, 5.0, {}),
            (self.username, "m2", 0, 4.5, {}),
        ]
        self.algo.test.return_value = predictions
        
        results = run_model(
            self.username,
            self.algo,
            self.user_data,
            self.sample_movie_list,
            movie_data=self.movie_data,
            num_recommendations=5,
            genre_blacklist=None
        )
        
        movie_ids = [r["movie_id"] for r in results]
        self.assertIn("m3", movie_ids)

if __name__ == '__main__':
    unittest.main()
