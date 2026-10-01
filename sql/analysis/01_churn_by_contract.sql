-- Churn rate and revenue split by contract and internet product
SELECT
    contract_type,
    internet_service,
    COUNT(*)                                   AS customers,
    SUM(churn)                                 AS churned,
    ROUND(100.0 * AVG(churn), 2)               AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2)             AS avg_monthly_charge,
    ROUND(SUM(monthly_charges), 2)             AS mrr,
    ROUND(SUM(CASE WHEN churn = 1 THEN monthly_charges ELSE 0 END), 2) AS mrr_lost
FROM vw_customer_360
GROUP BY contract_type, internet_service
ORDER BY churn_rate_pct DESC;
