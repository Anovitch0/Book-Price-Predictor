"""
scrape_books_bs.py
===================

This script scrapes book data from the demonstration site ``books.toscrape.com``
using ``requests`` and ``BeautifulSoup``. It collects product information
such as title, price, star rating, availability, category, UPC, product
description, and other metadata. The final dataset is written to a CSV file
in the ``data/`` directory.

Usage:
    python scrape_books_bs.py

The script will fetch all catalogue pages (50 pages in total) and follow
each product link to collect detailed information. A progress bar is
displayed during scraping. At the end, a summary of the dataset is shown
and the CSV file is saved.

Note:
    This script uses the demonstration website ``books.toscrape.com`` which
    contains 1000 books specifically for scraping practice. No credentials or
    authentication are required, and scraping is explicitly permitted by the
    site maintainers.
"""

import csv
import os
import re
from typing import List, Dict, Optional, Tuple

import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Configuration

BASE_URL = "https://books.toscrape.com/"
# The catalogue pages are located at ``catalogue/page-<n>.html``. The first
# page is ``index.html``. We determine the number of pages dynamically by
# following the "next" button rather than hard‑coding 50.

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CSV_FILE = os.path.join(DATA_DIR, "books_data_bs.csv")


def get_soup(url: str) -> BeautifulSoup:
    """Fetch the page at ``url`` and return a parsed ``BeautifulSoup`` object.

    Args:
        url: The URL to fetch.

    Returns:
        BeautifulSoup instance representing the HTML page.
    """
    # Use a desktop browser User‑Agent string to avoid potential 403 Forbidden errors
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def parse_rating(rating_str: str) -> int:
    """Convert a rating string (e.g., ``'Three'``) into an integer 1–5.

    The site encodes ratings as CSS classes like ``"star-rating Three"``.

    Args:
        rating_str: The textual rating ("One", "Two", etc.).

    Returns:
        Integer rating from 1 to 5.
    """
    ratings_map = {
        "Zero": 0,
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five": 5,
    }
    return ratings_map.get(rating_str.strip(), 0)


def parse_product_page(url: str) -> Dict[str, Optional[str]]:
    """Parse a product detail page and return a dictionary of attributes.

    Args:
        url: Absolute URL to the product page.

    Returns:
        A dictionary containing product metadata.
    """
    soup = get_soup(url)

    # Breadcrumb navigation: Home > Books > Category > Book Title
    breadcrumb = soup.select("ul.breadcrumb li a")
    category = breadcrumb[2].get_text(strip=True) if len(breadcrumb) > 2 else None

    # Product information table
    table = soup.find("table", class_="table table-striped")
    product_info = {}
    if table:
        rows = table.find_all("tr")
        for row in rows:
            header = row.th.get_text(strip=True)
            value = row.td.get_text(strip=True)
            product_info[header] = value

    # Description
    description_tag = soup.find("div", id="product_description")
    if description_tag:
        description = description_tag.find_next_sibling("p").get_text(strip=True)
    else:
        description = None

    return {
        "upc": product_info.get("UPC"),
        "product_type": product_info.get("Product Type"),
        "price_excl_tax": product_info.get("Price (excl. tax)"),
        "price_incl_tax": product_info.get("Price (incl. tax)"),
        "tax": product_info.get("Tax"),
        "availability_detail": product_info.get("Availability"),
        "num_reviews": product_info.get("Number of reviews"),
        "category": category,
        "description": description,
    }


def scrape_books() -> pd.DataFrame:
    """Scrape all book listings and return a pandas DataFrame.

    This function iterates through all catalogue pages, extracts basic
    information for each book, follows the detail link for additional metadata,
    and assembles a DataFrame.

    Returns:
        DataFrame containing one row per book with detailed metadata.
    """
    books_data: List[Dict[str, Optional[str]]] = []

    # Start at the main catalogue page
    page_url = BASE_URL + "index.html"
    page_number = 1

    with tqdm(total=1000, desc="Scraping books", unit="book") as pbar:
        while True:
            soup = get_soup(page_url)
            product_list = soup.find_all("article", class_="product_pod")
            for product in product_list:
                # Basic info
                title_tag = product.find("h3").find("a")
                title = title_tag["title"] if title_tag.has_attr("title") else title_tag.get_text(strip=True)
                relative_link = title_tag["href"]
                # Normalize relative link: some links start with "../"
                product_url = BASE_URL + "catalogue/" + relative_link.replace("../../../", "").replace("../", "")
                price_text = product.find("p", class_="price_color").get_text(strip=True)
                price = float(re.sub(r"[^0-9.]+", "", price_text))
                availability_text = product.find("p", class_="instock availability").get_text(strip=True)
                star_tag = product.find("p", class_=re.compile("star-rating"))
                star_classes = star_tag["class"]
                # The second class is the rating word
                rating_word = star_classes[1] if len(star_classes) > 1 else "Zero"
                rating = parse_rating(rating_word)

                # Fetch detailed product page
                details = parse_product_page(product_url)

                book_info = {
                    "title": title,
                    "price": price,
                    "availability": availability_text,
                    "rating": rating,
                }
                book_info.update(details)
                books_data.append(book_info)
                pbar.update(1)

            # Find next page
            next_link = soup.find("li", class_="next")
            if next_link and next_link.find("a"):
                next_href = next_link.find("a")["href"]
                # Build the URL: index.html for first page, then catalogue/page-<n>.html
                # On the first page (index.html), next_href is "catalogue/page-2.html".
                if "catalogue/" in next_href:
                    page_url = BASE_URL + next_href
                else:
                    # If next_href is just page-<n>.html, prefix catalogue/
                    page_url = BASE_URL + "catalogue/" + next_href
                page_number += 1
            else:
                # No more pages
                break

    df = pd.DataFrame(books_data)
    return df


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Starting scraping using BeautifulSoup...")
    df = scrape_books()
    print(f"Scraped {len(df)} books.")
    # Save to CSV
    df.to_csv(CSV_FILE, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"Data saved to {CSV_FILE}")


if __name__ == "__main__":
    main()