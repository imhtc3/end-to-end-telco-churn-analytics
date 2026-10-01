# Data dictionary

## Source

IBM *Telco Customer Churn* sample (`WA_Fn-UseC_-Telco-Customer-Churn.csv`), 7,043 customers of a
fictional California telco, one row per customer, 21 columns. Published by IBM as a Cognos Analytics
sample and widely mirrored (Kaggle: `blastchar/telco-customer-churn`).

| Raw column | Clean column | Type | Notes |
|---|---|---|---|
| customerID | customer_id | text | primary key |
| gender | gender | Male / Female | |
| SeniorCitizen | senior_citizen | 0/1 | 65+ |
| Partner | partner | 0/1 | |
| Dependents | dependents | 0/1 | |
| tenure | tenure | int | months with the company at snapshot |
| PhoneService | phone_service | 0/1 | |
| MultipleLines | multiple_lines | Yes / No / No phone service | |
| InternetService | internet_service | DSL / Fiber optic / No | |
| OnlineSecurity … StreamingMovies | online_security … streaming_movies | Yes / No / No internet service | six add-ons |
| Contract | contract | Month-to-month / One year / Two year | |
| PaperlessBilling | paperless_billing | 0/1 | |
| PaymentMethod | payment_method | 4 levels | two of them automatic |
| MonthlyCharges | monthly_charges | float | current bill |
| TotalCharges | total_charges | float | blank for 11 customers with tenure 0 → imputed 0 |
| Churn | churn | 0/1 | left within the last month |

## Engineered features

| Feature | Definition | Why |
|---|---|---|
| n_addons | count of the six add-ons = "Yes" | engagement / stickiness |
| n_services | phone + extra lines + internet + add-ons | breadth of relationship |
| has_protection_bundle | ≥ 2 of security, tech support, device protection | the add-ons that matter most |
| auto_pay | payment method is automatic | friction to leave |
| contract_months | 1 / 12 / 24 | ordinal contract length |
| tenure_band | 0-6m, 7-12m, 13-24m, 25-48m, 49-60m, 61m+ | cohort reporting |
| avg_historic_charge | total_charges / tenure | what they paid on average |
| charge_drift | monthly_charges − avg_historic_charge | bill increases over time |
| charge_per_service | monthly_charges / n_services | perceived value for money |
| fiber_no_support | fiber customer without tech support | known pain point |
| price_vs_peers *(SQL)* | monthly charge − mean for same internet × contract | over-paying relative to similar customers |
| tenure_pct_in_contract *(SQL)* | percent rank of tenure within contract type | reporting only, not used in models |

## Warehouse tables

`dim_customer`, `dim_contract`, `dim_payment`, `dim_internet`, `fact_subscription`,
`bridge_customer_service`, `model_scores`; views `vw_customer_360` and `vw_feature_mart`.
DDL in `sql/schema.sql` and `sql/views.sql`.
