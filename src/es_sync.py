"""Copy product embeddings from PostgreSQL into an Elasticsearch vector index.

    python -m src.es_sync                                   # HNSW m=16, ef_construction=100
    python -m src.es_sync --index-type int8_hnsw --m 32 --force-merge --verify

Requires the sandbox started with ``--profile es``. Each run recreates the
index, so different HNSW / quantization settings can be compared on the same
data (Phase 6 builds a benchmark table out of exactly this).
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from . import config, dataset

INDEX_TYPES = ("hnsw", "int8_hnsw", "int4_hnsw", "bbq_hnsw", "flat")


def mappings(index_type: str, m: int, ef_construction: int) -> dict:
    index_options: dict = {"type": index_type}
    if index_type.endswith("hnsw"):
        index_options.update(m=m, ef_construction=ef_construction)
    return {
        "properties": {
            "tenant_id": {"type": "integer"},
            "category": {"type": "keyword"},
            "price": {"type": "float"},
            "stock": {"type": "integer"},
            "embedding": {
                "type": "dense_vector",
                "dims": dataset.EMBEDDING_DIM,
                "index": True,
                "similarity": "l2_norm",  # matches pgvector's <-> operator
                "index_options": index_options,
            },
        }
    }


def sync(pg_dsn: str, es_url: str, index: str, index_type: str, m: int, ef_construction: int,
         batch_size: int = 2000, force_merge: bool = False) -> int:
    import psycopg
    from elasticsearch import Elasticsearch, helpers

    es = Elasticsearch(es_url, request_timeout=300)
    if es.indices.exists(index=index):
        es.indices.delete(index=index)
    es.indices.create(
        index=index,
        mappings=mappings(index_type, m, ef_construction),
        settings={"number_of_shards": 1, "number_of_replicas": 0, "refresh_interval": "-1"},
    )

    def actions(cur):
        for product_id, tenant_id, category, price, stock, embedding in cur:
            yield {
                "_index": index,
                "_id": product_id,
                "_source": {"tenant_id": tenant_id, "category": category, "price": price,
                            "stock": stock, "embedding": json.loads(embedding)},
            }

    t0 = time.perf_counter()
    # A named (server-side) cursor streams rows instead of loading them all.
    with psycopg.connect(pg_dsn) as conn, conn.cursor(name="drift_products_export") as cur:
        cur.itersize = batch_size
        cur.execute("SELECT id, tenant_id, category, price::float8, stock, embedding::text "
                    "FROM products ORDER BY id")
        indexed, errors = helpers.bulk(es, actions(cur), chunk_size=batch_size, raise_on_error=False)
    for err in errors[:5]:
        print(f"  bulk error: {err}", file=sys.stderr)
    print(f"indexed {indexed} products in {time.perf_counter() - t0:.1f}s ({len(errors)} errors)")

    es.indices.put_settings(index=index, settings={"refresh_interval": "1s"})
    es.indices.refresh(index=index)
    if force_merge:
        t0 = time.perf_counter()
        es.indices.forcemerge(index=index, max_num_segments=1)
        print(f"force-merged to 1 segment in {time.perf_counter() - t0:.1f}s")
    return indexed


def verify(pg_dsn: str, es_url: str, index: str, k: int = 10) -> None:
    """Compare ES kNN results with an exact pgvector scan for one product."""
    import psycopg
    from elasticsearch import Elasticsearch

    with psycopg.connect(pg_dsn) as conn:
        product_id, embedding = conn.execute(
            "SELECT id, embedding::text FROM products ORDER BY id LIMIT 1").fetchone()
        # Force an exact scan even if a pgvector ANN index exists.
        conn.execute("SET LOCAL enable_indexscan = off")
        exact = [r[0] for r in conn.execute(
            "SELECT id FROM products ORDER BY embedding <-> %s::vector LIMIT %s", (embedding, k))]

    es = Elasticsearch(es_url)
    hits = es.search(index=index, knn={"field": "embedding", "query_vector": json.loads(embedding),
                                       "k": k, "num_candidates": 10 * k},
                     source=False, size=k)["hits"]["hits"]
    approx = [int(h["_id"]) for h in hits]
    recall = len(set(exact) & set(approx)) / k
    print(f"product {product_id}: recall@{k} vs pgvector = {recall:.2f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.es_sync", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dsn", default=config.pg_dsn())
    parser.add_argument("--es-url", default=config.es_url())
    parser.add_argument("--index", default="products")
    parser.add_argument("--index-type", choices=INDEX_TYPES, default="hnsw")
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--force-merge", action="store_true")
    parser.add_argument("--verify", action="store_true", help="spot-check recall against pgvector")
    args = parser.parse_args(argv)

    sync(args.dsn, args.es_url, args.index, args.index_type, args.m, args.ef_construction,
         batch_size=args.batch_size, force_merge=args.force_merge)
    if args.verify:
        verify(args.dsn, args.es_url, args.index)
    return 0


if __name__ == "__main__":
    sys.exit(main())
