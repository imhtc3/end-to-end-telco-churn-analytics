-- Warehouse layout for the churn project (SQLite dialect).
-- One row per customer in the fact table since the source is a single snapshot.

PRAGMA foreign_keys = ON;

DROP VIEW  IF EXISTS vw_feature_mart;
DROP VIEW  IF EXISTS vw_customer_360;
DROP TABLE IF EXISTS model_scores;
DROP TABLE IF EXISTS bridge_customer_service;
DROP TABLE IF EXISTS fact_subscription;
DROP TABLE IF EXISTS dim_customer;
DROP TABLE IF EXISTS dim_contract;
DROP TABLE IF EXISTS dim_payment;
DROP TABLE IF EXISTS dim_internet;

CREATE TABLE dim_contract (
    contract_key     INTEGER PRIMARY KEY,
    contract_type    TEXT    NOT NULL UNIQUE,
    contract_months  INTEGER NOT NULL
);

CREATE TABLE dim_payment (
    payment_key      INTEGER PRIMARY KEY,
    payment_method   TEXT    NOT NULL UNIQUE,
    auto_pay         INTEGER NOT NULL CHECK (auto_pay IN (0, 1))
);

CREATE TABLE dim_internet (
    internet_key     INTEGER PRIMARY KEY,
    internet_service TEXT    NOT NULL UNIQUE
);

CREATE TABLE dim_customer (
    customer_id      TEXT PRIMARY KEY,
    gender           TEXT    NOT NULL,
    senior_citizen   INTEGER NOT NULL,
    partner          INTEGER NOT NULL,
    dependents       INTEGER NOT NULL,
    family           INTEGER NOT NULL
);

CREATE TABLE fact_subscription (
    customer_id        TEXT PRIMARY KEY REFERENCES dim_customer(customer_id),
    contract_key       INTEGER NOT NULL REFERENCES dim_contract(contract_key),
    payment_key        INTEGER NOT NULL REFERENCES dim_payment(payment_key),
    internet_key       INTEGER NOT NULL REFERENCES dim_internet(internet_key),
    tenure             INTEGER NOT NULL,
    tenure_band        TEXT    NOT NULL,
    phone_service      INTEGER NOT NULL,
    multiple_lines     TEXT    NOT NULL,
    paperless_billing  INTEGER NOT NULL,
    n_addons           INTEGER NOT NULL,
    n_services         INTEGER NOT NULL,
    monthly_charges    REAL    NOT NULL,
    total_charges      REAL    NOT NULL,
    charge_drift       REAL,
    churn              INTEGER NOT NULL CHECK (churn IN (0, 1))
);

-- add-on services in long format, easier to slice in BI tools
CREATE TABLE bridge_customer_service (
    customer_id   TEXT NOT NULL REFERENCES dim_customer(customer_id),
    service_name  TEXT NOT NULL,
    status        TEXT NOT NULL,
    PRIMARY KEY (customer_id, service_name)
);

CREATE TABLE model_scores (
    customer_id        TEXT NOT NULL REFERENCES dim_customer(customer_id),
    model_name         TEXT NOT NULL,
    churn_probability  REAL NOT NULL,
    risk_tier          TEXT NOT NULL,
    scored_at          TEXT NOT NULL,
    PRIMARY KEY (customer_id, model_name)
);

CREATE INDEX ix_fact_contract ON fact_subscription(contract_key);
CREATE INDEX ix_fact_churn    ON fact_subscription(churn);
CREATE INDEX ix_bridge_svc    ON bridge_customer_service(service_name, status);
