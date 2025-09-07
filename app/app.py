"""
app.py
======

This Streamlit application provides an interactive interface for exploring
the synthetic books dataset and predicting book prices based on user
input. It loads a pre‑trained machine learning model (trained in
``train_model.py``) and uses it to generate price estimates from
features such as category, rating, availability, and description
length.

Key features:

* **Dataset explorer** – View the first few rows of the dataset.
* **Price distribution** – Inspect average prices by category via a bar chart.
* **Price predictor** – Input your own book metadata and obtain an
  estimated price.
* **About** – Learn more about the project and its synthetic nature.

To run the app locally, navigate to the ``data_science_project``
directory and execute:

    streamlit run app/app.py

Note: ``streamlit`` must be installed in your Python environment. If you
encounter an import error, install the package with ``pip install
streamlit``.
"""

import os
import joblib
import pandas as pd

try:
    import streamlit as st
except ImportError:
    raise ImportError(
        "Streamlit is not installed. Please install it with `pip install streamlit` "
        "before running this app."
    )


# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "scraped_books_local.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "price_model.pkl")


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the books dataset into a DataFrame."""
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    """Load the trained machine learning pipeline."""
    return joblib.load(MODEL_PATH)


def main() -> None:
    st.set_page_config(page_title="Book Price Predictor", page_icon="📚", layout="centered")
    st.title("📚 Synthetic Book Price Predictor")
    st.markdown(
        "This app demonstrates a simple machine learning model built on a synthetic "
        "dataset of 5,000 books. Use the sidebar to explore the data and predict prices."
    )

    # Sidebar navigation
    menu = ["Dataset", "Predict", "About"]
    choice = st.sidebar.selectbox("Navigation", menu)

    # Load resources
    df = load_data()
    model = load_model()

    if choice == "Dataset":
        st.header("Dataset Overview")
        st.write(
            "Below is a preview of the first 20 rows of the scraped synthetic dataset. "
            "Each record includes the book's category, rating (1–5), stock availability, "
            "a short description, and the price."
        )
        st.dataframe(df.head(20))
        # Price distribution chart
        st.subheader("Average Price by Category")
        avg_price = df.groupby('category')['price'].mean().sort_values(ascending=False)
        st.bar_chart(avg_price)

    elif choice == "Predict":
        st.header("Predict a Book Price")
        st.write(
            "Enter the details of your hypothetical book below. The model will estimate "
            "a price based on patterns learned from the synthetic dataset."
        )
        # Input widgets
        categories = sorted(df['category'].unique())
        category = st.selectbox("Category", categories)
        rating = st.slider("Rating (1–5)", min_value=1, max_value=5, value=3)
        availability = st.number_input(
            "Stock Availability", min_value=1, max_value=50, value=10, step=1
        )
        description = st.text_area(
            "Brief Description", value="An engaging tale of adventure and discovery."
        )
        # When the user clicks predict
        if st.button("Predict Price"):
            # Compute description length
            desc_length = len(description.split()) if description else 0
            input_df = pd.DataFrame([
                {
                    'category': category,
                    'rating': rating,
                    'availability': availability,
                    'description_length': desc_length,
                }
            ])
            # Predict
            predicted_price = model.predict(input_df)[0]
            st.success(f"Estimated Price: £{predicted_price:.2f}")

    else:  # About
        st.header("About This Project")
        st.write(
            "This project was created to demonstrate the end‑to‑end workflow of "
            "data collection (scraping), data processing, machine learning, and "
            "web application development. Since internet access is restricted in the "
            "execution environment, the dataset used here is synthetic. It was "
            "generated programmatically to mimic an online bookstore and then "
            "scraped from local HTML files using BeautifulSoup."
        )
        st.write(
            "The machine learning model is a RandomForestRegressor trained to predict "
            "book prices based on category, rating, stock availability and the length "
            "of the book description. The Streamlit app provides an interface to "
            "explore the dataset and generate predictions."
        )


if __name__ == '__main__':
    main()