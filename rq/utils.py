"""
@brief General utilities for making requests and working with the movie data.
"""

# Constants
from pathlib import Path

HOME_PATH = Path(__file__).resolve().parent.parent
DATASET_PATH = HOME_PATH / "datasets"

MOVIES_PATH = DATASET_PATH / "2024_us_movie_ids.csv"
REVIEW_PATH = DATASET_PATH / "reviews.csv"
SENTIMENT_PATH = DATASET_PATH / "sentiment.csv"

DEV_PATH = HOME_PATH / ".dev"

def _load_api_key() -> str:
    # check the file exists
    if not DEV_PATH.exists():
        raise FileNotFoundError(f".dev file not found at {DEV_PATH}. Create and put your TMDB key in.")
    
    # read through
    with DEV_PATH.open("r") as f:
        for raw in f:
            line = raw.strip()
            
            if "=" in line:
                # clean kv pair
                key, val = line.split("=", 1)
                key = key.strip().upper()
                val = val.strip().strip('"').strip("'")
                
                # check if it's the key
                if key == "TMDB_API_KEY":
                    return val
    
    # failed
    raise ValueError("API key not found in .dev file. Expected a line like: TMDB_API_KEY=<your_key>")

API_KEY = _load_api_key()

# Utility Functions
import requests
from typing import Any

def get_reviews(movie_id: int) -> list[dict[str, Any]]:
    """Gets all reviews for the given movie.
    """
    
    # setup call
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/reviews"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    # track all reviews
    reviews = list()
    
    # get response
    res = requests.get(
        url,
        headers=headers
    )
    res.raise_for_status()
    first_page = res.json()
    
    # check the total number of pages
    npages = first_page.get("total_pages", 1)
    
    # accumulate all results
    reviews.extend(first_page.get("results", []))
    
    for page_num in range(2, npages + 1):
        # make req
        res = requests.get(
            url,
            params={"page": page_num},
            headers=headers
        )
        res.raise_for_status()
        res = res.json()
        
        # append
        reviews.extend(res.get("results", []))
    
    # give back as a list of dicts
    return reviews

def transform_review(review_pkg: dict) -> dict[str, Any]:
    """Transforms a single review into its core components for use in creating
    the dataset.
    """
    
    # extract needed fields with defaults
    rating = review_pkg.get("author_details", {}).get("rating")
    
    return {
        "name": review_pkg.get("author", ""),
        "rating": float(rating) if rating else -1.0,
        "review": review_pkg.get("content", ""),
        "timestamp": review_pkg.get("created_at", "")
    }

def accumulate_reviews(reviews: list[dict]) -> dict:
    """Converts a list of individual reviews into a single dictionary of lists.
    """
    
    # accumulate and return
    return {
        "name": [r["name"] for r in reviews],
        "rating": [r["rating"] for r in reviews],
        "review": [r["review"] for r in reviews],
        "timestamp": [r["timestamp"] for r in reviews]
    }


# script for testing
if __name__ == "__main__":
    print(get_reviews(1097549))
