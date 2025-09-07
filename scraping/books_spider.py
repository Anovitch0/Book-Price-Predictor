"""
books_spider.py
================

This module defines a simple Scrapy spider that demonstrates how to
structure a Scrapy project for scraping book data. Due to the network
restrictions in the current environment, the spider does not perform
real HTTP requests; instead, it reads book data from a local CSV file
and yields Scrapy items as if they were scraped from the web. The goal
is to illustrate the typical components of a Scrapy spider—requests,
parsing methods, item classes—and how they integrate with pipelines.

To run this spider in a normal Scrapy project (outside this restricted
environment), you would need to install Scrapy and configure a project
with ``scrapy startproject``. Then place this file in the ``spiders``
folder and run:

    scrapy crawl books

In this example, the spider reads the file ``data/books_dataset.csv``
located in the parent directories. It yields items containing the title,
price, rating, and category of each book. A pipeline could then be
implemented to process or store these items in a database.
"""

import os
import csv
from typing import Iterator

import scrapy


class BookItem(scrapy.Item):
    title = scrapy.Field()
    price = scrapy.Field()
    rating = scrapy.Field()
    category = scrapy.Field()


class BooksSpider(scrapy.Spider):
    name = "books"
    custom_settings = {
        "LOG_LEVEL": "INFO",
    }

    def start_requests(self) -> Iterator[scrapy.Request]:
        """Generate the initial request.

        In a normal spider, this method would yield requests to the
        catalogue pages of a website. Here, we simply yield a dummy
        request to the ``parse_csv`` callback, passing the file path via
        the ``meta`` attribute.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(os.path.dirname(script_dir), "data", "books_dataset.csv")
        # Use scrapy's dummy Request with a file:// URL; the callback will
        # simply read the file. The actual URL is irrelevant here.
        yield scrapy.Request(
            url=f"file://{data_path}",
            callback=self.parse_csv,
            dont_filter=True,
            meta={"csv_path": data_path},
        )

    def parse_csv(self, response: scrapy.http.Response) -> Iterator[BookItem]:
        """Read a local CSV file and yield Scrapy items.

        Args:
            response: A Scrapy Response object. In this context, the
                ``response.url`` will refer to a file:// URL.

        Yields:
            BookItem objects for each row in the CSV.
        """
        csv_path = response.meta["csv_path"]
        self.logger.info(f"Loading data from {csv_path}")
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                item = BookItem()
                item["title"] = row["Book Name"]
                item["price"] = float(row["Price"])
                item["rating"] = int(row["Rate"])
                item["category"] = row["Category"]
                yield item