"""
@brief Script for generating the sentiment dataset for RQ1 and RQ2.2
"""

# -- Env Setup -- #
# importss
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

import json
from pathlib import Path

from rq.utils import *

# model params
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else (
        "mps" if torch.mps.is_available() else "cpu"
    )
)

# -- Helpers -- #
def prepare_comparison(reviews: pd.DataFrame) -> pd.DataFrame:
    """Prepares for the statistical comparison for review scores with sentiment
    inferred scores.
        - Converts "# stars" into an integer
        - Accounts for different scale (1 - 5 vs 1 - 10)
        - Creates new indicator column, correct or not --> "score_match"
    """
    
    # convert into numeric columns
    pred_scores = reviews["predicted_rating"].str.split().str[0].astype(float)
    
    # scale the review scores; NaN reviews will have NaN indicator values
    human_scores = reviews["rating"].astype(float)
    human_scores.loc[human_scores <= 0.0] = pd.NA
    human_scores = (human_scores % 2) + (human_scores // 2.0) # 1,2 --> 1 star, 3,4 --> 2 star, etc.
    
    # compare to the predicted scores
    reviews["score_match"] = pred_scores == human_scores
    reviews["unknown_target"] = human_scores.isna()
    reviews["converted_rating"] = human_scores
    reviews["converted_prediction"] = pred_scores
    
    # clean reviews to get rid of newlines
    reviews["review"] = reviews["review"].str.replace("\n", " ")
    
    # export
    reviews.to_csv(CLEANED_PATH, index=False)
    return reviews


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
    
    # load
    reviews = pd.read_csv(SENTIMENT_PATH)
    
    # ready for comparison
    reviews = prepare_comparison(reviews)
    
    ### PLOTS ###
    # drop rows with missing values
    df = reviews.dropna(subset=["converted_rating", "converted_prediction"]).copy()

    # convert to numeric
    x = df["converted_rating"].astype(float)
    y = df["converted_prediction"].astype(float)
    conf = df["prediction_confidence"].astype(float)

    # grid
    sns.set_style("darkgrid")
    plt.figure(figsize=(8, 6), dpi=300)

    # scatter, color with prediction confidence
    sc = plt.scatter(x, y, c=conf, cmap="viridis", alpha=0.8, s=40, edgecolor="k", linewidth=0.2)
    cbar = plt.colorbar(sc)
    cbar.set_label("prediction_confidence")

    # line of best fit
    sns.regplot(x=x, y=y, scatter=False, color="black", line_kws={"linewidth": 2})

    # annotate and format
    plt.xlabel("Human Rating")
    plt.ylabel("Predicted Rating (Adjusted)")
    plt.tight_layout()
    plt.savefig(FIG_PATH / "rq1_scatter.png")
    plt.clf()
    
    # prediction diffs plot
    diff = x - y
    plt.figure(figsize=(8, 6), dpi=300)

    sns.kdeplot(diff, fill=True, cmap="viridis", bw_method="scott", alpha=0.9)
    plt.axvline(0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Rating Difference (human - predicted)")
    plt.ylabel("Density")
    plt.title("Rating Differences Distribution")
    plt.tight_layout()
    plt.savefig(FIG_PATH / "rq1_kde_diff.png")
    plt.clf()
    
    ### STATISTICS ###
    # R^2
    ss_res = ((df.converted_rating - df.converted_prediction) ** 2).sum()
    ss_tot = ((df.converted_rating - df.converted_rating.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    
    metrics = {
        "Accuracy": reviews["score_match"].mean(skipna=True),
        "Correlation": df.converted_rating.corr(df.converted_prediction),
        "R2": r2,
    }
    
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)


# -- Main -- #
if __name__ == "__main__":
    # gen_review_dataset()
    # gen_sentiment_dataset()
    compare_review_scores()

