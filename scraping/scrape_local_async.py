"""
scrape_local_async.py
=====================

This script demonstrates how to scrape data from multiple local HTML files
concurrently using Python's ``asyncio`` framework. Although reading files
from disk is generally fast, this example showcases how you can structure
asynchronous scraping code that could easily be adapted to fetch data
over the network using ``aiohttp`` or similar libraries.

The script uses ``asyncio.to_thread`` to run the synchronous parsing
function in separate threads. This approach is simple and avoids
introducing additional dependencies such as ``aiofiles``.

Usage:
    python scrape_local_async.py

The output CSV will be saved to ``data/scraped_books_local_async.csv``.
"""

import asyncio
import csv
import os
from typing import List, Dict

import pandas as pd

from scrape_local_html import parse_html_file, HTML_DIR, DATA_DIR  # reuse parsing logic

OUTPUT_CSV_ASYNC = os.path.join(DATA_DIR, "scraped_books_local_async.csv")


async def parse_file_async(file_path: str) -> List[Dict[str, str]]:
    """Asynchronously parse a single HTML file by running the synchronous
    ``parse_html_file`` function in a separate thread.

    Args:
        file_path: Path to the HTML file.

    Returns:
        A list of record dictionaries.
    """
    return await asyncio.to_thread(parse_html_file, file_path)


async def scrape_all_pages_async(html_dir: str) -> pd.DataFrame:
    """Scrape all HTML files in ``html_dir`` concurrently and return a DataFrame."""
    tasks = []
    for filename in sorted(os.listdir(html_dir)):
        if not filename.endswith('.html'):
            continue
        file_path = os.path.join(html_dir, filename)
        tasks.append(parse_file_async(file_path))
    all_results: List[List[Dict[str, str]]] = await asyncio.gather(*tasks)
    # Flatten list of lists
    records: List[Dict[str, str]] = [rec for sublist in all_results for rec in sublist]
    df = pd.DataFrame(records)
    # Convert numeric columns
    numeric_cols = ['id', 'rating', 'price', 'availability']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def main() -> None:
    df = asyncio.run(scrape_all_pages_async(HTML_DIR))
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(OUTPUT_CSV_ASYNC, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"Asynchronously scraped {len(df)} records from local HTML pages.")
    print(f"Data saved to {OUTPUT_CSV_ASYNC}")


if __name__ == '__main__':
    main()