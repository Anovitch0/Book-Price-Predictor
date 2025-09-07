"""
generate_synthetic_books.py
===========================

This script generates a synthetic dataset of book records and writes
both a CSV file and one or more HTML files representing the data. The
dataset is intended to emulate the structure of data you might scrape
from an e‑commerce bookstore (similar to ``books.toscrape.com``) while
allowing you to run everything locally without hitting external
resources. Each record includes a title, category, price, rating, stock
availability, a short description, and other attributes commonly found
on product pages.

The HTML output is useful for demonstrating web scraping with
``BeautifulSoup``. Splitting the data across multiple HTML files makes
it possible to illustrate asynchronous scraping with ``aiohttp`` or
``asyncio``, as each file can be parsed concurrently.

Usage:
    python generate_synthetic_books.py

Generated files:
    data/synthetic_books.csv           - CSV file containing all records.
    scraping/local/pages/page_1.html   - First HTML page (1000 rows).
    scraping/local/pages/page_2.html   - ...

The number of pages and rows per page can be adjusted via constants.
"""

import csv
import os
import random
import string
from typing import List

import pandas as pd


# Configuration
NUM_RECORDS = 5000
ROWS_PER_PAGE = 1000

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
LOCAL_HTML_DIR = os.path.join(os.path.dirname(__file__), "local", "pages")
CSV_FILE = os.path.join(DATA_DIR, "synthetic_books.csv")


def random_word(length: int) -> str:
    """Generate a random lowercase word of a given length."""
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))


def random_sentence(min_words: int = 8, max_words: int = 20) -> str:
    """Generate a random sentence with a variable number of words."""
    num_words = random.randint(min_words, max_words)
    words = [random_word(random.randint(3, 10)) for _ in range(num_words)]
    sentence = ' '.join(words).capitalize() + '.'
    return sentence


def generate_dataset(num_records: int) -> pd.DataFrame:
    """Generate a synthetic dataset of book records as a DataFrame."""
    categories = [
        'Travel', 'Mystery', 'Historical Fiction', 'Sequential Art', 'Classics',
        'Philosophy', 'Romance', 'Womens Fiction', 'Fiction', 'Childrens',
        'Religion', 'Nonfiction', 'Music', 'Default', 'Science Fiction',
        'Sports and Games', 'Fantasy', 'New Adult', 'Young Adult', 'Science',
        'Poetry', 'Paranormal', 'Art', 'Psychology', 'Autobiography',
        'Parenting', 'Adult Fiction', 'Humor', 'Horror', 'History'
    ]

    adjectives = [
        'Amazing', 'Incredible', 'Mysterious', 'Fantastic', 'Enchanting',
        'Thrilling', 'Majestic', 'Curious', 'Serene', 'Vivid', 'Silent',
        'Forgotten', 'Golden', 'Hidden', 'Ancient', 'Brave', 'Clever',
        'Whispering', 'Radiant', 'Shadowy'
    ]
    nouns = [
        'Journey', 'Legacy', 'Secret', 'Chronicle', 'Saga', 'Quest', 'Tale',
        'Legend', 'Story', 'Odyssey', 'Mystery', 'Adventure', 'Dream', 'Voice',
        'Whisper', 'Echo', 'Promise', 'Empire', 'Garden', 'Sky'
    ]

    data: List[dict] = []
    for i in range(num_records):
        title = f"{random.choice(adjectives)} {random.choice(nouns)} {i+1}"
        category = random.choice(categories)
        rating = random.randint(1, 5)
        # Price correlated with rating but with noise
        base_price = rating * 5
        price = round(base_price + random.uniform(5, 25), 2)
        availability = random.randint(1, 20)
        description = random_sentence()
        data.append({
            'id': i + 1,
            'title': title,
            'category': category,
            'rating': rating,
            'price': price,
            'availability': availability,
            'description': description,
        })
    df = pd.DataFrame(data)
    return df


def save_csv(df: pd.DataFrame, path: str) -> None:
    """Save the DataFrame to a CSV file with quoted strings."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_NONNUMERIC)


def save_html_pages(df: pd.DataFrame, directory: str, rows_per_page: int) -> None:
    """Split the DataFrame into multiple HTML pages and write them to disk.

    Each page contains a table with ``rows_per_page`` rows. The HTML files
    include a minimal structure (``<html>``/``<body>``) so they can be parsed
    easily by BeautifulSoup. A basic navigation section allows linking
    between pages (although it's not used by the scraper).

    Args:
        df: The full dataset.
        directory: Directory where HTML pages will be saved.
        rows_per_page: Number of rows to include per page.
    """
    os.makedirs(directory, exist_ok=True)
    total_records = len(df)
    num_pages = (total_records + rows_per_page - 1) // rows_per_page
    for page_idx in range(num_pages):
        start = page_idx * rows_per_page
        end = min(start + rows_per_page, total_records)
        page_df = df.iloc[start:end]
        page_number = page_idx + 1
        # Build navigation links
        nav_links = []
        if page_idx > 0:
            nav_links.append(f'<a href="page_{page_idx}.html">Previous</a>')
        if page_idx < num_pages - 1:
            nav_links.append(f'<a href="page_{page_idx + 2}.html">Next</a>')
        nav_html = ' | '.join(nav_links)
        table_html = page_df.to_html(index=False, classes='table table-striped', border=0)
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Synthetic Books Page {page_number}</title>
</head>
<body>
    <h1>Synthetic Books Page {page_number}</h1>
    <div class="navigation">{nav_html}</div>
    {table_html}
</body>
</html>
"""
        file_path = os.path.join(directory, f"page_{page_number}.html")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)


def main() -> None:
    df = generate_dataset(NUM_RECORDS)
    save_csv(df, CSV_FILE)
    save_html_pages(df, LOCAL_HTML_DIR, ROWS_PER_PAGE)
    print(f"Generated {len(df)} synthetic book records.")
    print(f"CSV saved to {CSV_FILE}")
    print(f"HTML pages saved to {LOCAL_HTML_DIR}")


if __name__ == '__main__':
    main()