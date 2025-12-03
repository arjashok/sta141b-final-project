"""
@brief Script for generating the sentiment dataset for RQ1 and RQ2.2
"""

# -- Env Setup -- #
# importss
import torch
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

from pathlib import Path

from rq.utils import *

# model params
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
DEVICE = "mps" #0 if torch.cuda.is_available() else -1

# -- Scripts -- #
def gen_review_dataset() -> None:
    """Generates the dataset with just the reviews.
    """
    
    # load the movie ids
    movies = pd.read_csv(MOVIES_PATH)["movie_id"]
    
    # for each, request the reviews for it
    reviews = list()
    
    for movie in tqdm(movies):
        # get and transform reviews
        r = get_reviews(movie)
        r = list(map(transform_review, r))
        
        # accumulate into one dictionary/frame
        r = accumulate_reviews(r)
        
        # compress into a dataframe and append
        r = pd.DataFrame(r)
        r["movie_id"] = movie
        reviews.append(r)
    
    # concat & save
    reviews = pd.concat(reviews, ignore_index=True)
    reviews.to_csv(REVIEW_PATH, index=False)

def gen_sentiment_dataset() -> None:
    """Generates a new column for the review dataset with the model-guess for
    the review scores.
    """
    
    # setup model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    pipe = pipeline(
        "text-classification", model=model, tokenizer=tokenizer, device=DEVICE,
        truncation=True
    )
    
    # load the review dataset
    reviews = pd.read_csv(REVIEW_PATH)
    
    # generate the sentiment scores for each review    
    sentiment_packages = pipe(reviews["review"].tolist())
    ratings = list(map(lambda x: x["label"], sentiment_packages))
    confs = list(map(lambda x: x["score"], sentiment_packages))
    
    # add columns and export
    reviews["predicted_rating"] = ratings
    reviews["prediction_confidence"] = confs
    reviews.to_csv(SENTIMENT_PATH, index=False)
    
def compare_review_scores():
    """Generates the figures and statistics comparing the human-chosen score vs
    the model-predicted score.
    """
    
    pass


# -- Main -- #
if __name__ == "__main__":
    gen_review_dataset()
    gen_sentiment_dataset()

