-- Synthetic multi-tenant commerce schema.
--
-- Tables deliberately ship with PRIMARY KEYS ONLY: choosing secondary
-- indexes is the agent's job. No foreign keys either, to keep loading fast.
-- Recreating the schema drops every table (and therefore every index).

CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS order_items, orders, events, products, customers, tenants, drift_meta CASCADE;

-- Deterministic hash of a row id into [0, 1). Mirrors src/dataset.py:unit_hash().
CREATE OR REPLACE FUNCTION drift_unit_hash(row_id bigint, salt int)
RETURNS double precision
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
    SELECT mod(row_id * 2654435761 + salt * 2246822519, 4294967296)::double precision
           / 4294967296::double precision
$$;

-- Owning tenant of a row (Zipf-like skew, tenant 1 is the largest).
-- Mirrors src/dataset.py:tenant_of().
CREATE OR REPLACE FUNCTION drift_tenant_of(row_id bigint, n_tenants int, salt int)
RETURNS int
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
    SELECT 1 + floor(n_tenants * (h * h * h))::int
    FROM (SELECT drift_unit_hash(row_id, salt) AS h) s
$$;

CREATE TABLE tenants (
    id          int PRIMARY KEY,
    name        text NOT NULL,
    plan        text NOT NULL,
    region      text NOT NULL,
    created_at  timestamptz NOT NULL
);

CREATE TABLE customers (
    id          bigint PRIMARY KEY,
    tenant_id   int NOT NULL,
    email       text NOT NULL,          -- mixed case on purpose: lower(email) lookups
    country     text NOT NULL,
    signup_at   timestamptz NOT NULL,
    is_active   boolean NOT NULL
);

CREATE TABLE products (
    id          bigint PRIMARY KEY,
    tenant_id   int NOT NULL,
    sku         text NOT NULL,
    category    text NOT NULL,
    price       numeric(10, 2) NOT NULL,
    stock       int NOT NULL,
    created_at  timestamptz NOT NULL,
    embedding   vector(64) NOT NULL     -- keep in sync with dataset.EMBEDDING_DIM
);

CREATE TABLE orders (
    id          bigint PRIMARY KEY,
    tenant_id   int NOT NULL,
    customer_id bigint NOT NULL,
    status      text NOT NULL,
    total       numeric(12, 2) NOT NULL,
    created_at  timestamptz NOT NULL
);

CREATE TABLE order_items (
    order_id    bigint NOT NULL,
    line_no     int NOT NULL,
    product_id  bigint NOT NULL,
    quantity    int NOT NULL,
    unit_price  numeric(10, 2) NOT NULL,
    PRIMARY KEY (order_id, line_no)
);

CREATE TABLE events (
    id          bigint PRIMARY KEY,
    tenant_id   int NOT NULL,
    customer_id bigint NOT NULL,
    event_type  text NOT NULL,
    payload     jsonb NOT NULL,
    created_at  timestamptz NOT NULL
);

-- Records how the dataset was generated (read back by the workload runner).
CREATE TABLE drift_meta (
    key         text PRIMARY KEY,
    value       jsonb NOT NULL
);
