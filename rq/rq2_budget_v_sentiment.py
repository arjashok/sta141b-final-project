"""
@brief Script for comparing budget and sentiment.
"""

# -- Env Setup -- #
# importss
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import json
from pathlib import Path

from rq.utils import *


# -- Scripts -- #
def combine_datasets() -> pd.DataFrame:
    """Joins the budget data with the sentiment data.
    """
    
    # load the datasets
    budgets = pd.read_csv(BUDGET_PATH)
    reviews = pd.read_csv(CLEANED_PATH)
    
    # join along the movie ids
    combined = pd.merge(
        reviews, budgets, how="left", left_on="movie_id", right_on="id"
    )
    
    # export
    return combined

def compare_budget_v_sentiment() -> None:
    """Bar chart comparison of budget vs sentiment per movie.
    """
    
    # merge
    combined = combine_datasets()
    filtered = combined[combined["converted_prediction"].notna()]
    
    # find the average for the bar charts
    avg_budgets = filtered.groupby("budget")["converted_prediction"].mean()
    
    # find the distributions
    moviewise_budgets = filtered.groupby("movie_id")[["converted_prediction", "budget"]].mean()
    print(avg_budgets)
    print(moviewise_budgets)


# -- Main -- #
if __name__ == "__main__":
    compare_budget_v_sentiment()

