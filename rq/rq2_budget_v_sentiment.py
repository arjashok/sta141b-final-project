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

HIGH_BUDGET_THRESHOLD = 50_000_000


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
    filtered["high_budget"] = filtered["budget"] >= HIGH_BUDGET_THRESHOLD
    
    # find the average for the bar charts
    avg_budgets = filtered.groupby("high_budget")["converted_prediction"].mean().reset_index()
    
    # find the distributions
    moviewise_budgets = filtered.groupby("movie_id")[["converted_prediction", "budget", "high_budget"]].mean().reset_index()
    
    # plot 1
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(6, 4))

    avg_budgets["high_budget"] = avg_budgets["high_budget"].map({True: "High (>=50M)", False: "Low (<50M)"})
    ax = sns.barplot(x="high_budget", y="converted_prediction", data=avg_budgets, palette="viridis", hue="high_budget")
    
    ax.set_xlabel("Budget")
    ax.set_ylabel("Average Sentiment")
    ax.set_title("Average Sentiment by Budget Category")
    
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2.0, p.get_height()),
                    ha="center", va="bottom", fontsize=9)
    plt.tight_layout()

    outpath = FIG_PATH / "budget_vs_sentiment.png"
    plt.savefig(outpath, dpi=300)
    plt.close()
    
    # plot 2
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(8, 6))

    moviewise_budgets["budget_m"] = moviewise_budgets["budget"] / 1_000_000
    colors = {True: "tab:blue", False: "tab:pink"}
    labels = {True: "High (>=50M)", False: "Low (<50M)"}

    for hb in [True, False]:
        subset = moviewise_budgets[moviewise_budgets["high_budget"] == hb]
        sns.regplot(
            x="budget_m",
            y="converted_prediction",
            data=subset,
            scatter=True,
            color=colors[hb],
            label=labels[hb],
            line_kws={"linewidth": 2},
        )

    plt.xlabel("Budget (Millions USD)")
    plt.ylabel("Average Sentiment")
    plt.title("Movie-wise Sentiment vs Budget")
    plt.legend(title="Budget Category")
    plt.tight_layout()

    outpath = FIG_PATH / "budget_vs_sentiment_regressions.png"
    plt.savefig(outpath, dpi=300)
    plt.close()


# -- Main -- #
if __name__ == "__main__":
    compare_budget_v_sentiment()

