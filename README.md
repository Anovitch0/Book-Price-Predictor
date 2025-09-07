# Synthetic Book Price Prediction Project

## Overview

This repository contains a complete data science project that mimics the
process of scraping an online bookstore, cleaning the data, training a
machine learning model, and deploying a predictive web application.

Because internet access is disabled in the execution environment,
**the data used here is synthetic**. We programmatically generate a
dataset of 5,000 books with random titles, categories, ratings,
prices and descriptions. The dataset is then exported to HTML pages
that resemble product listing pages on an e‑commerce site. These
HTML pages are scraped using BeautifulSoup to produce the final data
frame used for modelling.

Despite the synthetic nature of the data, the project still covers
real‑world techniques: scraping (from HTML), feature engineering,
model training and evaluation, and web app development with
Streamlit.

## Directory Structure

```
data_science_project/
├── data/
│   ├── synthetic_books.csv            # Raw synthetic data
│   ├── scraped_books_local.csv        # Data scraped from local HTML pages
│   ├── scraped_books_local_async.csv  # Same as above, scraped asynchronously
├── scraping/
│   ├── generate_synthetic_books.py    # Generates synthetic data and HTML files
│   ├── scrape_local_html.py           # Scrapes HTML pages synchronously
│   ├── scrape_local_async.py          # Scrapes HTML pages concurrently
│   └── local/
│       └── pages/
│           ├── page_1.html
│           ├── ...
├── ml/
│   ├── train_model.py                # Trains RandomForestRegressor model
│   └── models/
│       ├── price_model.pkl           # Trained model pipeline
│       └── model_performance.md      # Evaluation report
├── app/
│   └── app.py                        # Streamlit app
├── README.md
└── requirements.txt
```

## How to Reproduce

1. **Generate the dataset and HTML pages** (already run in this repository):

   ```bash
   python scraping/generate_synthetic_books.py
   ```

2. **Scrape the data from local HTML** (creates `data/scraped_books_local.csv`):

   ```bash
   python scraping/scrape_local_html.py
   # Or run the asynchronous version:
   python scraping/scrape_local_async.py
   ```

3. **Train the machine learning model**:

   ```bash
   python ml/train_model.py
   ```

   The script will output evaluation metrics, save the trained model
   pipeline to `ml/models/price_model.pkl` and write a report to
   `ml/models/model_performance.md`.

4. **Run the Streamlit app**:

   ```bash
   streamlit run app/app.py
   ```

   This will launch a local web server where you can explore the
   dataset and make price predictions. Note that you need to install
   `streamlit` in your environment first (e.g. `pip install streamlit`).

## Dependencies

All required Python packages are listed in `requirements.txt`. To
install them into a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Limitations

* The data is synthetic and does not reflect real book prices or
  descriptions. It is intended purely for educational purposes.

* The Streamlit app code is included, but Streamlit itself may not be
  installed in the provided runtime. Install it manually to run the
  app.

