#!/usr/bin/env python3

import random
from datetime import datetime, timedelta

from elasticsearch import Elasticsearch

random.seed(42)

INDEX = "reviews"

es = Elasticsearch("http://localhost:9200")

STORES = [40, 41, 42, 43, 44]

PRODUCTS = list(range(1, 11))


# ---------------------------------------------------------------------
# Review templates
# ---------------------------------------------------------------------

POSITIVE = [
    "Friendly staff and quick checkout.",
    "Clean store and easy to find products.",
    "Everything I needed was in stock.",
    "Good prices and helpful employees.",
    "Checkout was fast.",
    "Nice shopping experience.",
    "Fresh products and clean shelves.",
    "Would shop here again.",
    "Store was well organized.",
    "Good selection of products.",
]

NEUTRAL = [
    "Average experience.",
    "Nothing special but everything was fine.",
    "Store was a little busy.",
    "Found what I needed.",
    "Lines were slightly long.",
    "Decent overall.",
]

PRODUCT_POSITIVE = [
    "The product quality was good.",
    "Happy with this purchase.",
    "Worth buying.",
    "Good value for the money.",
    "Exactly what I expected.",
]

OVERCHARGE = [
    "I was overcharged at checkout.",
    "The price at the register was much higher than the shelf price.",
    "Charged way more than expected.",
    "Receipt shows an incorrect price.",
    "Cashier charged me an absurd amount.",
    "Something was seriously wrong with the price.",
    "The register overcharged me.",
    "Very disappointed after being charged far too much.",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def random_timestamp_last_30_days():
    now = datetime.now()

    days = random.randint(0, 29)
    seconds = random.randint(0, 86399)

    return now - timedelta(days=days, seconds=seconds)


def weighted_rating():
    """
    Mostly 3-5 stars.
    """
    return random.choices(
        population=[1, 2, 3, 4, 5],
        weights=[2, 6, 25, 37, 30],
        k=1,
    )[0]


# ---------------------------------------------------------------------
# Review generation
# ---------------------------------------------------------------------

def generate_normal_reviews():
    docs = []

    for _ in range(150):

        rating = weighted_rating()

        if rating >= 4:
            text = random.choice(POSITIVE)
        elif rating == 3:
            text = random.choice(NEUTRAL)
        else:
            text = "Could have been better."

        # Around 25% are store-level reviews
        if random.random() < 0.25:
            product_id = None
        else:
            product_id = random.choice(PRODUCTS)

            if rating >= 4 and random.random() < 0.5:
                text += " " + random.choice(PRODUCT_POSITIVE)

        docs.append({
            "store_id": random.choice(STORES),
            "product_id": product_id,
            "rating": rating,
            "text": text,
            "ts": random_timestamp_last_30_days().isoformat(timespec="seconds"),
        })

    return docs


# ---------------------------------------------------------------------
# Planted anomaly
# ---------------------------------------------------------------------

def plant_overcharge_reviews(docs):
    """
    Reviews corresponding to the 100x price glitch.

    Same evening:
    yesterday between 17:30 and 18:30
    """

    base = (
        datetime.now()
        - timedelta(days=1)
    ).replace(
        hour=17,
        minute=30,
        second=0,
        microsecond=0,
    )

    for i in range(random.randint(5, 8)):

        ts = base + timedelta(
            minutes=random.randint(0, 60),
            seconds=random.randint(0, 59),
        )

        docs.append({
            "store_id": 42,
            "product_id": random.choice(PRODUCTS),
            "rating": 1,
            "text": random.choice(OVERCHARGE),
            "ts": ts.isoformat(timespec="seconds"),
        })


# ---------------------------------------------------------------------
# Elasticsearch
# ---------------------------------------------------------------------

def recreate_index():

    try:
        print(es.info())
    except Exception as e:
        print(repr(e))

    if es.indices.exists(index=INDEX):
        es.indices.delete(index=INDEX)

    es.indices.create(
        index=INDEX,
        mappings={
            "properties": {
                "store_id": {
                    "type": "integer"
                },
                "product_id": {
                    "type": "integer"
                },
                "rating": {
                    "type": "integer"
                },
                "text": {
                    "type": "text"
                },
                "ts": {
                    "type": "date"
                },
            }
        },
    )


def bulk_insert(docs):

    operations = []

    for doc in docs:
        operations.append({"index": {"_index": INDEX}})
        operations.append(doc)

    es.bulk(operations=operations, refresh=True)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    recreate_index()

    docs = generate_normal_reviews()

    plant_overcharge_reviews(docs)

    random.shuffle(docs)

    bulk_insert(docs)

    print(f"Inserted {len(docs)} reviews.")


if __name__ == "__main__":
    main()