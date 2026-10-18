from collections import Counter

from src import dataset
from src.dataset import DatasetSpec


def test_unit_hash_in_unit_interval():
    for i in range(1, 10_000):
        assert 0.0 <= dataset.unit_hash(i, 0) < 1.0


def test_tenant_of_range_and_skew():
    counts = Counter(dataset.tenant_of(i) for i in range(1, 100_001))
    assert min(counts) >= 1 and max(counts) <= dataset.N_TENANTS
    # Zipf-like: tenant 1 owns ~ (1/50)^(1/3) = 27% of rows.
    assert 0.22 < counts[1] / 100_000 < 0.32
    assert counts[1] > counts[2] > counts[dataset.N_TENANTS]


def test_salts_give_different_owners():
    same = sum(dataset.tenant_of(i, salt=dataset.CUSTOMER_SALT) == dataset.tenant_of(i, salt=dataset.PRODUCT_SALT)
               for i in range(1, 1001))
    assert same < 1000


def test_order_status_probabilities_sum_to_one():
    assert abs(sum(p for _, p in dataset.ORDER_STATUSES) - 1.0) < 1e-9


def test_spec_round_trip():
    spec = DatasetSpec(scale=0.1)
    assert DatasetSpec.from_dict(spec.to_dict()) == spec
    assert spec.n_orders == 100_000
