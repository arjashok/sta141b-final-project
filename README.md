# sta141b-final-project
Movie reviews, budget, and revenues scraping + analysis.

## Environment Setup
Clone the repo via:
```
git clone https://github.com/arjashok/sta141b-final-project.git ./group4_project/
cd group4_project/
```

Install all required packages. We highly recommend using `miniconda` or `uv` to
manage this!
```
pip install -r requirements
```

See the Layout section for info on how to run the notebooks/scripts.

## Layout
The codebase is structured as a package (`rq`) with all results outside of the
core python scripts/notebooks. Given the presence of notebooks for our results
generation, we recommend NOT installing `rq` as a package and instead running
notebooks as normal and scripts as follows:

```
python3 -m rq.rq1_sentiment
python3 -m rq.rq2_budget_v_sentiment
```

The above will work if the current working directory is the `/path/to/repo`.

