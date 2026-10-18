import json
import random

import pytest

from src import dataset, workload
from src.dataset import DatasetSpec

SPEC = DatasetSpec(scale=0.1)


def queries(scenario, n=2000, seed=1):
    return [r for r in workload.generate(scenario, n, SPEC, seed=seed) if r["kind"] == "query"]


def test_scenario_mixes_only_reference_known_templates():
    for phases in workload.SCENARIOS.values():
        for phase in phases:
            assert set(phase.mix) <= set(workload.TEMPLATES)


@pytest.mark.parametrize("name", sorted(workload.TEMPLATES))
def test_template_params_match_placeholders(name):
    template = workload.TEMPLATES[name]
    placeholders = set(workload.PLACEHOLDER_RE.findall(template.sql))
    sampler = workload.ParamSampler(SPEC, random.Random(0))
    for _ in range(20):
        params = template.sample(sampler)
        assert set(params) == placeholders
        json.dumps(params)  # must be serialisable into the log
        assert "%(" not in workload.render_sql(template.sql, params)


def test_sampled_customer_belongs_to_tenant():
    sampler = workload.ParamSampler(SPEC, random.Random(0))
    for _ in range(200):
        params = workload.TEMPLATES["orders_by_customer"].sample(sampler)
        assert dataset.tenant_of(params["customer_id"], SPEC.n_tenants, dataset.CUSTOMER_SALT) == params["tenant_id"]


def test_generation_is_deterministic():
    assert queries("full", 300, seed=7) == queries("full", 300, seed=7)
    assert queries("full", 300, seed=7) != queries("full", 300, seed=8)


def test_header_and_counts():
    records = list(workload.generate("oltp_to_analytics", 500, SPEC, seed=0))
    assert records[0]["kind"] == "header"
    assert records[0]["dataset"]["scale"] == 0.1
    assert len(records) == 501
    assert [r["seq"] for r in records[1:]] == list(range(500))


def test_oltp_to_analytics_drifts():
    recs = queries("oltp_to_analytics", 2000)
    first, last = recs[:700], recs[-500:]  # phase 1 is the first 40%
    assert {r["template"] for r in first} <= set(workload.OLTP_MIX)
    assert {r["template"] for r in last} <= set(workload.ANALYTICS_MIX)
    assert all(r["phase"] == "analytics" for r in last)


def test_tenant_onboarding_shifts_traffic_to_small_tenant():
    recs = queries("tenant_onboarding", 2000)
    hot = dataset.N_TENANTS

    def hot_share(rs):
        with_tenant = [r for r in rs if "tenant_id" in r["params"]]
        return sum(r["params"]["tenant_id"] == hot for r in with_tenant) / len(with_tenant)

    assert hot_share(recs[:1000]) < 0.05
    assert hot_share(recs[1000:]) > 0.45


def test_transition_blends_mixes():
    phases = workload.SCENARIOS["oltp_to_analytics"]
    mix, _, _ = workload.phase_state(phases, 1, 0.25)  # halfway through the transition
    assert mix["customer_by_email"] == pytest.approx(10)  # 0.5 * 20 + 0.5 * 0
    assert mix["top_products"] == pytest.approx(12.5)
    mix, _, _ = workload.phase_state(phases, 1, 0.9)
    assert mix == workload.ANALYTICS_MIX


def test_render_sql_escapes_quotes():
    assert workload.render_sql("SELECT %(a)s, %(b)s", {"a": "O'Brien", "b": 3}) == "SELECT 'O''Brien', 3"


def test_percentile():
    values = list(range(1, 101))
    assert workload.percentile(values, 50) == 50
    assert workload.percentile(values, 95) == 95
    assert workload.percentile([7.0], 95) == 7.0


def test_cli_generate(tmp_path):
    out = tmp_path / "w.jsonl"
    assert workload.main(["generate", "--scenario", "vector_launch", "-n", "50", "--scale", "0.1",
                          "--out", str(out)]) == 0
    header, records = workload.read_log(out)
    assert header["scenario"] == "vector_launch"
    assert len(records) == 50
