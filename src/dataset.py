"""Shared description of the synthetic multi-tenant dataset.

The seeding SQL (``src/seed.py``) and the workload generator
(``src/workload.py``) must agree on table sizes, value domains and which
tenant owns each customer/product id. Everything they share lives here.

Tenant ownership is a deterministic hash of the row id, implemented twice:
here in Python and as ``drift_tenant_of()`` in ``src/sql/schema.sql``. Both
use the same IEEE-754 operations so they give identical answers, which lets
the workload generator pick (tenant, customer) pairs that really exist
without querying the database.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

N_TENANTS = 50
EMBEDDING_DIM = 64  # keep in sync with vector(64) in src/sql/schema.sql

DATA_START = datetime(2024, 1, 1, tzinfo=timezone.utc)
DATA_END = datetime(2026, 1, 1, tzinfo=timezone.utc)

CATEGORIES = (
    "electronics", "books", "clothing", "home", "garden", "toys",
    "sports", "beauty", "grocery", "automotive", "music", "office",
)
COUNTRIES = ("US", "IN", "GB", "DE", "BR", "NP", "CA", "FR", "JP", "AU")
EVENT_TYPES = ("page_view", "search", "add_to_cart", "checkout", "login", "review")
EVENT_SOURCES = ("web", "ios", "android", "api")  # skewed toward the front

# (status, probability). The skew is intentional: rare statuses such as
# 'pending' are what make partial indexes worthwhile.
ORDER_STATUSES = (
    ("delivered", 0.70),
    ("shipped", 0.15),
    ("cancelled", 0.08),
    ("pending", 0.05),
    ("refunded", 0.02),
)

CUSTOMER_SALT = 0  # orders and events inherit their customer's tenant
PRODUCT_SALT = 1
CATEGORY_SALT = 2

_HASH_MULT = 2654435761  # Knuth multiplicative hash
_HASH_SALT_MULT = 2246822519  # large odd constant so each salt shifts the hash substantially
_HASH_MOD = 2**32


def unit_hash(row_id: int, salt: int = 0) -> float:
    """Deterministic hash of ``row_id`` into [0, 1). Mirrors ``drift_unit_hash``."""
    return ((row_id * _HASH_MULT + salt * _HASH_SALT_MULT) % _HASH_MOD) / float(_HASH_MOD)


def tenant_of(row_id: int, n_tenants: int = N_TENANTS, salt: int = CUSTOMER_SALT) -> int:
    """Owning tenant of a row. Zipf-like: tenant 1 is the largest (~27% of rows
    with 50 tenants), the highest-numbered tenant the smallest (<1%).
    Mirrors ``drift_tenant_of``."""
    h = unit_hash(row_id, salt)
    return 1 + int(n_tenants * (h * h * h))


def category_of(product_id: int) -> str:
    """Category of a product. Mirrors the drift_centroids join in src/seed.py."""
    return CATEGORIES[int(unit_hash(product_id, CATEGORY_SALT) * len(CATEGORIES))]


@dataclass(frozen=True)
class DatasetSpec:
    """Row counts for a given scale factor. Scale 1.0 is roughly 600 MB on disk."""

    scale: float = 1.0
    n_tenants: int = N_TENANTS

    @property
    def n_customers(self) -> int:
        return max(1_000, int(200_000 * self.scale))

    @property
    def n_products(self) -> int:
        return max(500, int(50_000 * self.scale))

    @property
    def n_orders(self) -> int:
        return max(5_000, int(1_000_000 * self.scale))

    @property
    def n_events(self) -> int:
        return max(5_000, int(2_000_000 * self.scale))

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "n_customers": self.n_customers,
            "n_products": self.n_products,
            "n_orders": self.n_orders,
            "n_events": self.n_events,
            "embedding_dim": EMBEDDING_DIM,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetSpec":
        return cls(scale=float(d["scale"]), n_tenants=int(d.get("n_tenants", N_TENANTS)))
