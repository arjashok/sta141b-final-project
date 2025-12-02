"""
@brief Script for generating the sentiment dataset for RQ1 and RQ2.2
"""

# -- Env Setup -- #
# imports
import torch
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

from pathlib import Path

# model params
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
DEVICE = 0 if torch.cuda.is_available() else -1

# -- Scripts -- #
def gen_review_dataset(movie_path: Path) -> None:
    """Generates the dataset with just the reviews

    Args:
        movie_path (Path): _description_
    """
    
    # load the movie ids

def gen_sentiment_dataset():
    """Generates a new column for the review dataset with the model-guess for
    the review scores.
    """
    
    # setup model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=DEVICE)
    
    
    
def compare_review_scores():
    """Generates the figures and statistics comparing the human-chosen score vs
    the model-predicted score.
    """
    
    pass


# -- Main -- #
if __name__ == "__main__":
    gen_

