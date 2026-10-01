# Power BI model

Source: CSV extracts in `data/powerbi/`, written by the `powerbi` pipeline step.
Queries live in `powerbi/queries.pq`, measures in `powerbi/measures.dax`, colours in `powerbi/theme.json`.

## Relationships

```
dim_contract (1) ──< fact_subscription >── (1) dim_payment
dim_internet (1) ──<        │          >── (1) dim_tenure_band
                            │
dim_customer (1) ─── (1) fact_subscription (1) ─── (1) fact_churn_score
dim_segment  (1) ─── (1)    │
                            └──< bridge_customer_service
```

| From | To | Cardinality | Filter direction |
|---|---|---|---|
| fact_subscription[contract_key] | dim_contract[contract_key] | *:1 | single |
| fact_subscription[payment_key] | dim_payment[payment_key] | *:1 | single |
| fact_subscription[internet_key] | dim_internet[internet_key] | *:1 | single |
| fact_subscription[tenure_band] | dim_tenure_band[tenure_band] | *:1 | single |
| fact_subscription[customer_id] | dim_customer[customer_id] | 1:1 | both |
| fact_subscription[customer_id] | dim_segment[customer_id] | 1:1 | both |
| fact_subscription[customer_id] | fact_churn_score[customer_id] | 1:1 | both |
| bridge_customer_service[customer_id] | fact_subscription[customer_id] | *:1 | single |

Sort-by-column: `dim_tenure_band[tenure_band]` by `band_order`, `fact_churn_score[risk_tier]` by `tier_order`.

## Report pages

**1. Overview**
KPI cards (Customers, Churn Rate, MRR, MRR Lost, ARPU Churned vs Retained) ·
churn rate by contract (bar) · churn by tenure band with cumulative MRR lost line ·
contract × internet matrix with `Churn Rate Colour` as background conditional formatting ·
slicers: gender, age group, household, payment method.

**2. Drivers**
Add-on adoption vs churn (bridge table, `service` on axis) ·
payment method × paperless billing matrix · monthly charges histogram split by status ·
decomposition tree on `Churn Rate` (contract → internet → payment → tenure band).

**3. Risk & Campaign**
Customers by risk tier (stacked by actual status) · `Expected MRR at Risk` by segment ·
what-if slicers for offer cost / acceptance / months retained / threshold driving
`Campaign Net Value` and `Campaign ROI` · table of active high-risk customers
sorted by probability × monthly charge (the call list).

**4. Segments**
Scatter of avg tenure vs ARPU per segment, bubble size = customers ·
segment profile table · churn rate by segment.
