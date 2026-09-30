"""Database initialization, relationships and read-only access."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError, OperationalError

from database.connection import DatabaseNotFoundError, get_engine
from database.init_db import NEVER_ORDERED, init_database
from database.schema import get_schema_description

EXPECTED_TABLES = {"customers", "products", "orders", "order_items"}


def test_tables_and_keys(ro_engine) -> None:
    insp = inspect(ro_engine)
    assert set(insp.get_table_names()) == EXPECTED_TABLES
    assert insp.get_pk_constraint("orders")["constrained_columns"] == ["order_id"]

    fks = {
        (t, fk["constrained_columns"][0], fk["referred_table"])
        for t in EXPECTED_TABLES
        for fk in insp.get_foreign_keys(t)
    }
    assert fks == {
        ("orders", "customer_id", "customers"),
        ("order_items", "order_id", "orders"),
        ("order_items", "product_id", "products"),
    }


def test_sample_data_shape(ro_engine) -> None:
    with ro_engine.connect() as conn:
        counts = {t: conn.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar_one()
                  for t in EXPECTED_TABLES}
        statuses = set(conn.execute(text("SELECT DISTINCT status FROM orders")).scalars())
        cities = conn.execute(text("SELECT COUNT(DISTINCT city) FROM customers")).scalar_one()
        categories = conn.execute(text("SELECT COUNT(DISTINCT category) FROM products")).scalar_one()
        months = conn.execute(
            text("SELECT COUNT(DISTINCT strftime('%Y-%m', order_date)) FROM orders")
        ).scalar_one()
        never_ordered = set(conn.execute(text(
            "SELECT p.product_name FROM products p "
            "LEFT JOIN order_items oi ON oi.product_id = p.product_id "
            "WHERE oi.order_item_id IS NULL"
        )).scalars())
        orphans = conn.execute(text(
            "SELECT COUNT(*) FROM order_items oi LEFT JOIN orders o USING (order_id) "
            "WHERE o.order_id IS NULL"
        )).scalar_one()

    assert counts["customers"] >= 40 and counts["products"] >= 20
    assert counts["orders"] >= 300 and counts["order_items"] > counts["orders"]
    assert statuses == {"completed", "cancelled", "pending"}
    assert cities >= 5 and categories >= 4 and months == 12
    assert never_ordered == NEVER_ORDERED
    assert orphans == 0


def test_init_is_idempotent_and_reset_rebuilds(tmp_path: Path) -> None:
    db = tmp_path / "idem.db"
    first = init_database(db)
    second = init_database(db)  # must not duplicate rows
    assert first == second

    engine = get_engine(db, read_only=False)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM order_items WHERE order_item_id = 1"))
    engine.dispose()
    assert init_database(db)["order_items"] == first["order_items"] - 1
    assert init_database(db, reset=True) == first  # deterministic rebuild


def test_foreign_keys_are_enforced(tmp_path: Path) -> None:
    db = tmp_path / "fk.db"
    init_database(db)
    engine = get_engine(db, read_only=False)
    try:
        with pytest.raises(IntegrityError):
            with engine.begin() as conn:
                conn.execute(text(
                    "INSERT INTO orders (order_id, customer_id, order_date, status) "
                    "VALUES (99999, 99999, '2025-01-01', 'completed')"
                ))
        with pytest.raises(IntegrityError):  # CHECK constraint on status
            with engine.begin() as conn:
                conn.execute(text(
                    "INSERT INTO orders (order_id, customer_id, order_date, status) "
                    "VALUES (99998, 1, '2025-01-01', 'shipped')"
                ))
    finally:
        engine.dispose()


def test_read_only_engine_refuses_writes(ro_engine) -> None:
    with pytest.raises(OperationalError):
        with ro_engine.begin() as conn:
            conn.execute(text("DELETE FROM order_items"))
    with ro_engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM order_items")).scalar_one() > 0


def test_read_only_engine_requires_existing_file(tmp_path: Path) -> None:
    with pytest.raises(DatabaseNotFoundError):
        get_engine(tmp_path / "missing.db", read_only=True)
    assert not (tmp_path / "missing.db").exists()


def test_schema_description_lists_tables_relationships_and_values(ro_engine) -> None:
    schema = get_schema_description(ro_engine)
    for table in EXPECTED_TABLES:
        assert f"Table {table}" in schema
    assert "FOREIGN KEY customer_id -> customers.customer_id" in schema
    assert "'completed'" in schema and "'cancelled'" in schema and "'pending'" in schema
    assert "order_date range: 2025-01-" in schema
