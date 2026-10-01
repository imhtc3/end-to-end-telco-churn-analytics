CREATE VIEW vw_customer_360 AS
SELECT
    f.customer_id,
    c.gender,
    c.senior_citizen,
    c.partner,
    c.dependents,
    c.family,
    ct.contract_type,
    ct.contract_months,
    p.payment_method,
    p.auto_pay,
    i.internet_service,
    f.tenure,
    f.tenure_band,
    f.phone_service,
    f.multiple_lines,
    f.paperless_billing,
    f.n_addons,
    f.n_services,
    f.monthly_charges,
    f.total_charges,
    f.charge_drift,
    f.churn
FROM fact_subscription f
JOIN dim_customer c  ON c.customer_id   = f.customer_id
JOIN dim_contract ct ON ct.contract_key = f.contract_key
JOIN dim_payment  p  ON p.payment_key   = f.payment_key
JOIN dim_internet i  ON i.internet_key  = f.internet_key;


-- Modelling input. Add-ons are pivoted back to wide format, plus a couple of
-- features that are easier to express relative to the whole base in SQL.
CREATE VIEW vw_feature_mart AS
WITH addons AS (
    SELECT
        customer_id,
        MAX(CASE WHEN service_name = 'online_security'   THEN status END) AS online_security,
        MAX(CASE WHEN service_name = 'online_backup'     THEN status END) AS online_backup,
        MAX(CASE WHEN service_name = 'device_protection' THEN status END) AS device_protection,
        MAX(CASE WHEN service_name = 'tech_support'      THEN status END) AS tech_support,
        MAX(CASE WHEN service_name = 'streaming_tv'      THEN status END) AS streaming_tv,
        MAX(CASE WHEN service_name = 'streaming_movies'  THEN status END) AS streaming_movies
    FROM bridge_customer_service
    GROUP BY customer_id
),
peer AS (
    -- how expensive is this customer relative to others on the same product
    SELECT
        customer_id,
        monthly_charges
          - AVG(monthly_charges) OVER (PARTITION BY internet_service, contract_type) AS price_vs_peers,
        PERCENT_RANK() OVER (PARTITION BY contract_type ORDER BY tenure)             AS tenure_pct_in_contract
    FROM vw_customer_360
)
SELECT
    v.*,
    a.online_security,
    a.online_backup,
    a.device_protection,
    a.tech_support,
    a.streaming_tv,
    a.streaming_movies,
    ROUND(pr.price_vs_peers, 4)          AS price_vs_peers,
    ROUND(pr.tenure_pct_in_contract, 4)  AS tenure_pct_in_contract
FROM vw_customer_360 v
JOIN addons a ON a.customer_id  = v.customer_id
JOIN peer  pr ON pr.customer_id = v.customer_id;
