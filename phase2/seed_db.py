#!/usr/bin/env python3

import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)

DB_PATH = Path(__file__).parent / "retail.db"

# ---------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------

STORES = [
    (40, "Downtown", "New York"),
    (41, "Riverside", "Chicago"),
    (42, "Central Plaza", "Seattle"),
    (43, "Mall West", "Austin"),
    (44, "Harbor Point", "Boston"),
]

PRODUCTS = [
    (1, "Milk", "Dairy", 2.99),
    (2, "Bread", "Bakery", 3.49),
    (3, "Eggs", "Dairy", 4.99),
    (4, "Coffee", "Beverages", 8.99),
    (5, "Tea", "Beverages", 5.49),
    (6, "Rice", "Pantry", 6.99),
    (7, "Pasta", "Pantry", 2.49),
    (8, "Apples", "Produce", 1.79),
    (9, "Chicken", "Meat", 9.99),
    (10, "Soap", "Household", 3.99),
]

PRODUCT_LOOKUP = {p[0]: p for p in PRODUCTS}


# ---------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------

def create_tables(conn):
    cur = conn.cursor()

    cur.executescript("""
    DROP TABLE IF EXISTS transactions;
    DROP TABLE IF EXISTS products;
    DROP TABLE IF EXISTS stores;

    CREATE TABLE stores (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        city TEXT NOT NULL
    );

    CREATE TABLE products (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        unit_price REAL NOT NULL
    );

    CREATE TABLE transactions (
        id INTEGER PRIMARY KEY,
        store_id INTEGER NOT NULL,
        product_id INTEGER NOT NULL,
        quantity INTEGER NOT NULL,
        unit_price REAL NOT NULL,
        ts TEXT NOT NULL,
        FOREIGN KEY(store_id) REFERENCES stores(id),
        FOREIGN KEY(product_id) REFERENCES products(id)
    );
    """)

    cur.executemany(
        "INSERT INTO stores VALUES (?, ?, ?)",
        STORES,
    )

    cur.executemany(
        "INSERT INTO products VALUES (?, ?, ?, ?)",
        PRODUCTS,
    )

    conn.commit()


# ---------------------------------------------------------------------
# Transaction generation
# ---------------------------------------------------------------------

def random_timestamp_last_30_days():
    now = datetime.now()

    days_back = random.randint(0, 29)
    seconds = random.randint(0, 86399)

    return now - timedelta(days=days_back, seconds=seconds)


def generate_normal_transactions():
    rows = []

    transaction_id = 1

    for _ in range(2000):
        store = random.choice(STORES)[0]
        product = random.choice(PRODUCTS)

        qty = random.randint(1, 5)

        ts = random_timestamp_last_30_days()

        rows.append((
            transaction_id,
            store,
            product[0],
            qty,
            product[3],
            ts.isoformat(timespec="seconds"),
        ))

        transaction_id += 1

    return rows, transaction_id


# ---------------------------------------------------------------------
# Planted anomalies
# ---------------------------------------------------------------------

def plant_duplicate_burst(rows, next_id):
    """
    Same transaction repeated 30 times
    within one minute yesterday.
    """

    now = datetime.now()

    base_time = (
        now
        - timedelta(days=1)
    ).replace(hour=14, minute=15, second=0, microsecond=0)

    product = PRODUCT_LOOKUP[4]      # Coffee
    qty = 2

    for i in range(30):
        ts = base_time + timedelta(seconds=i)

        rows.append((
            next_id,
            42,
            product[0],
            qty,
            product[3],
            ts.isoformat(timespec="seconds"),
        ))
        next_id += 1

    return next_id


def plant_price_glitch(rows, next_id):
    """
    Price inflated by 100x.
    """

    now = datetime.now()

    base_time = (
        now
        - timedelta(days=1)
    ).replace(hour=17, minute=30, second=0, microsecond=0)

    for i in range(10):
        product = random.choice(PRODUCTS)
        qty = random.randint(1, 3)

        ts = base_time + timedelta(minutes=i)

        rows.append((
            next_id,
            42,
            product[0],
            qty,
            product[3] * 100,
            ts.isoformat(timespec="seconds"),
        ))

        next_id += 1

    return next_id

def shuffle_and_reassign_ids(rows):
    """
    Randomize row order so anomalies are scattered
    throughout the ID space, then assign new IDs.
    """
    random.shuffle(rows)

    new_rows = []

    for new_id, row in enumerate(rows, start=1):
        _, store_id, product_id, qty, price, ts = row

        new_rows.append((
            new_id,
            store_id,
            product_id,
            qty,
            price,
            ts,
        ))

    return new_rows

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    conn = sqlite3.connect(DB_PATH)

    create_tables(conn)

    transactions, next_id = generate_normal_transactions()

    next_id = plant_duplicate_burst(transactions, next_id)
    next_id = plant_price_glitch(transactions, next_id)

    # NEW
    transactions = shuffle_and_reassign_ids(transactions)

    conn.executemany(
        """
        INSERT INTO transactions
        (id, store_id, product_id, quantity, unit_price, ts)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        transactions,
    )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    main()