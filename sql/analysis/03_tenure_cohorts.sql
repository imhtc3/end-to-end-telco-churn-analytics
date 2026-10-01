-- Churn by tenure band. Bands are ordered by their lower bound, not alphabetically.
SELECT
    tenure_band,
    MIN(tenure)                         AS min_tenure,
    COUNT(*)                            AS customers,
    ROUND(100.0 * AVG(churn), 2)        AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2)      AS avg_monthly_charge,
    ROUND(AVG(n_services), 2)           AS avg_services,
    ROUND(100.0 * AVG(CASE WHEN contract_type = 'Month-to-month' THEN 1.0 ELSE 0 END), 1) AS pct_month_to_month
FROM vw_customer_360
GROUP BY tenure_band
ORDER BY min_tenure;
