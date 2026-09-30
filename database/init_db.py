"""Create and seed the QueryPilot sample sales database.

Usage (from the project root):

    python -m database.init_db           # create tables + seed only if empty
    python -m database.init_db --reset   # drop everything and rebuild

The data is generated with a fixed random seed, so every build produces the
exact same rows. Seeding is skipped when data already exists, which makes the
default command safe to run repeatedly without duplicating records.
"""

from __future__ import annotations

import argparse
import logging
import random
from datetime import date, timedelta
from pathlib import Path

from sqlalchemy import (
    CheckConstraint,
    Column,
    Date,
    Engine,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    func,
    insert,
    select,
)

from config import get_settings
from database.connection import get_engine

logger = logging.getLogger(__name__)

SEED = 42
ORDER_STATUSES = ("completed", "cancelled", "pending")

metadata = MetaData()

customers = Table(
    "customers",
    metadata,
    Column("customer_id", Integer, primary_key=True),
    Column("customer_name", String(100), nullable=False),
    Column("city", String(60), nullable=False),
    Column("signup_date", Date, nullable=False),
)

products = Table(
    "products",
    metadata,
    Column("product_id", Integer, primary_key=True),
    Column("product_name", String(120), nullable=False, unique=True),
    Column("category", String(60), nullable=False),
    Column("unit_price", Float, nullable=False),
    CheckConstraint("unit_price > 0", name="ck_products_unit_price_positive"),
)

orders = Table(
    "orders",
    metadata,
    Column("order_id", Integer, primary_key=True),
    Column("customer_id", Integer, ForeignKey("customers.customer_id"), nullable=False),
    Column("order_date", Date, nullable=False),
    Column("status", String(20), nullable=False),
    CheckConstraint(
        "status IN ('completed', 'cancelled', 'pending')", name="ck_orders_status_valid"
    ),
)

order_items = Table(
    "order_items",
    metadata,
    Column("order_item_id", Integer, primary_key=True),
    Column("order_id", Integer, ForeignKey("orders.order_id"), nullable=False),
    Column("product_id", Integer, ForeignKey("products.product_id"), nullable=False),
    Column("quantity", Integer, nullable=False),
    Column("unit_price", Float, nullable=False),
    CheckConstraint("quantity > 0", name="ck_order_items_quantity_positive"),
    CheckConstraint("unit_price > 0", name="ck_order_items_unit_price_positive"),
)

# --------------------------------------------------------------------------- #
# Source data for the generator
# --------------------------------------------------------------------------- #

# (city, relative weight) – weights make the customer distribution uneven,
# which gives "which city has the most customers?" a meaningful answer.
CITIES: list[tuple[str, int]] = [
    ("Mumbai", 9),
    ("Delhi", 8),
    ("Bengaluru", 7),
    ("Hyderabad", 5),
    ("Pune", 5),
    ("Chennai", 4),
    ("Kolkata", 3),
    ("Jaipur", 2),
]

FIRST_NAMES = [
    "Aarav", "Priya", "Rohan", "Ananya", "Vikram", "Sneha", "Arjun", "Kavya",
    "Rahul", "Meera", "Karan", "Isha", "Aditya", "Pooja", "Siddharth", "Neha",
    "Nikhil", "Riya", "Varun", "Divya", "Amit", "Shreya", "Manish", "Tanvi",
    "Harsh", "Aisha", "Kunal", "Nisha", "Yash", "Simran",
]
LAST_NAMES = [
    "Sharma", "Patel", "Iyer", "Reddy", "Gupta", "Nair", "Singh", "Mehta",
    "Kapoor", "Joshi", "Rao", "Das", "Chopra", "Kulkarni", "Banerjee", "Menon",
]

# (product_name, category, unit_price)
PRODUCTS: list[tuple[str, str, float]] = [
    ("Wireless Earbuds", "Electronics", 2499.00),
    ("Bluetooth Speaker", "Electronics", 3299.00),
    ("Smartwatch", "Electronics", 8999.00),
    ("USB-C Charger 65W", "Electronics", 1499.00),
    ("Mechanical Keyboard", "Electronics", 4599.00),
    ("Smart Ring", "Electronics", 19999.00),
    ("Non-stick Frying Pan", "Home & Kitchen", 1299.00),
    ("Electric Kettle", "Home & Kitchen", 1799.00),
    ("Air Fryer", "Home & Kitchen", 6499.00),
    ("Cotton Bedsheet Set", "Home & Kitchen", 1599.00),
    ("Espresso Machine Pro", "Home & Kitchen", 24999.00),
    ("The Pragmatic Programmer", "Books", 899.00),
    ("Atomic Habits", "Books", 499.00),
    ("Clean Code", "Books", 749.00),
    ("Data Science Handbook", "Books", 1199.00),
    ("Yoga Mat", "Sports", 999.00),
    ("Adjustable Dumbbells", "Sports", 7499.00),
    ("Running Shoes", "Sports", 4299.00),
    ("Badminton Racket", "Sports", 1899.00),
    ("Rowing Machine", "Sports", 32999.00),
    ("Denim Jacket", "Clothing", 2799.00),
    ("Cotton T-Shirt", "Clothing", 599.00),
    ("Formal Shirt", "Clothing", 1499.00),
    ("Winter Hoodie", "Clothing", 1999.00),
    ("Silk Saree", "Clothing", 5499.00),
]

# Products intentionally never sold, so "which products have never been
# ordered?" has a non-empty answer that must be found through a SQL anti-join.
NEVER_ORDERED = {"Smart Ring", "Espresso Machine Pro", "Rowing Machine"}

NUM_CUSTOMERS = 50
NUM_ORDERS = 420
ORDERS_START = date(2025, 1, 1)
ORDERS_END = date(2025, 12, 31)
# Relative order volume per month (Jan..Dec): festive-season spike in Oct-Dec.
MONTH_WEIGHTS = [6, 5, 6, 6, 7, 7, 7, 8, 9, 12, 14, 13]


def _random_date(rng: random.Random, start: date, end: date) -> date:
    return start + timedelta(days=rng.randint(0, (end - start).days))


def _order_date(rng: random.Random, not_before: date) -> date:
    month = rng.choices(range(1, 13), weights=MONTH_WEIGHTS)[0]
    month_start = date(2025, month, 1)
    month_end = (date(2025, month + 1, 1) - timedelta(days=1)) if month < 12 else ORDERS_END
    start = max(month_start, not_before)
    if start > month_end:  # customer signed up after this month; use their signup period
        start, month_end = not_before, ORDERS_END
    return _random_date(rng, start, month_end)


def _order_status(rng: random.Random, order_date: date) -> str:
    # Recent orders may still be pending; older ones are completed or cancelled.
    if (ORDERS_END - order_date).days <= 45:
        return rng.choices(ORDER_STATUSES, weights=[50, 10, 40])[0]
    return rng.choices(ORDER_STATUSES[:2], weights=[85, 15])[0]


def generate_sample_data(seed: int = SEED) -> dict[str, list[dict]]:
    """Build deterministic sample rows for every table."""
    rng = random.Random(seed)

    city_names = [c for c, _ in CITIES]
    city_weights = [w for _, w in CITIES]
    all_names = [f"{f} {l}" for f in FIRST_NAMES for l in LAST_NAMES]
    customer_names = rng.sample(all_names, NUM_CUSTOMERS)

    customer_rows = []
    for i, name in enumerate(customer_names, start=1):
        customer_rows.append(
            {
                "customer_id": i,
                "customer_name": name,
                "city": rng.choices(city_names, weights=city_weights)[0],
                "signup_date": _random_date(rng, date(2024, 1, 1), date(2025, 9, 30)),
            }
        )

    product_rows = [
        {"product_id": i, "product_name": n, "category": c, "unit_price": p}
        for i, (n, c, p) in enumerate(PRODUCTS, start=1)
    ]
    sellable = [p for p in product_rows if p["product_name"] not in NEVER_ORDERED]
    # Cheaper products sell more often.
    sell_weights = [1.0 / (p["unit_price"] ** 0.5) for p in sellable]

    # Some customers are much more active than others (top-customer questions).
    customer_activity = [rng.paretovariate(1.5) for _ in customer_rows]

    order_rows: list[dict] = []
    item_rows: list[dict] = []
    for order_id in range(1, NUM_ORDERS + 1):
        customer = rng.choices(customer_rows, weights=customer_activity)[0]
        order_date = _order_date(rng, max(customer["signup_date"], ORDERS_START))
        order_rows.append(
            {
                "order_id": order_id,
                "customer_id": customer["customer_id"],
                "order_date": order_date,
                "status": _order_status(rng, order_date),
            }
        )
        n_items = rng.choices([1, 2, 3, 4], weights=[45, 30, 17, 8])[0]
        chosen: list[dict] = []
        while len(chosen) < n_items:
            product = rng.choices(sellable, weights=sell_weights)[0]
            if product not in chosen:
                chosen.append(product)
        for product in chosen:
            max_qty = 5 if product["unit_price"] < 1000 else 2
            # Occasional 10% promotional discount, so order_items.unit_price
            # (price actually paid) can differ from products.unit_price (list price).
            discount = 0.9 if rng.random() < 0.15 else 1.0
            item_rows.append(
                {
                    "order_item_id": len(item_rows) + 1,
                    "order_id": order_id,
                    "product_id": product["product_id"],
                    "quantity": rng.randint(1, max_qty),
                    "unit_price": round(product["unit_price"] * discount, 2),
                }
            )

    return {
        "customers": customer_rows,
        "products": product_rows,
        "orders": order_rows,
        "order_items": item_rows,
    }


def init_database(db_path: Path | str, reset: bool = False) -> dict[str, int]:
    """Create tables and seed them if empty. Returns row counts per table."""
    engine: Engine = get_engine(db_path, read_only=False)
    try:
        if reset:
            logger.info("Dropping all tables")
            metadata.drop_all(engine)
        metadata.create_all(engine)  # no-op for tables that already exist

        with engine.begin() as conn:  # single transaction: all-or-nothing seed
            existing = conn.execute(select(func.count()).select_from(customers)).scalar_one()
            if existing:
                logger.info("Database already contains data; skipping seed")
            else:
                data = generate_sample_data()
                # Insert parents before children so FK checks pass.
                for table in (customers, products, orders, order_items):
                    conn.execute(insert(table), data[table.name])
                logger.info("Seeded sample data")

        with engine.connect() as conn:
            return {
                t.name: conn.execute(select(func.count()).select_from(t)).scalar_one()
                for t in metadata.sorted_tables
            }
    finally:
        engine.dispose()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Initialize the QueryPilot sample database.")
    parser.add_argument("--reset", action="store_true", help="drop all tables and rebuild")
    parser.add_argument("--db-path", type=Path, default=None, help="override the database path")
    args = parser.parse_args()

    db_path = args.db_path or get_settings().db_path
    counts = init_database(db_path, reset=args.reset)
    print(f"Database ready at: {Path(db_path).resolve()}")
    for table, count in counts.items():
        print(f"  {table:<12} {count:>5} rows")


if __name__ == "__main__":
    main()
