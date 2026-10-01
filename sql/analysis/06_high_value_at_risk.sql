-- Top-quartile spenders who churned, grouped by what they had.
-- Used to size the "premium save desk" idea.
WITH ranked AS (
    SELECT *,
           NTILE(4) OVER (ORDER BY monthly_charges) AS spend_quartile
    FROM vw_customer_360
)
SELECT
    contract_type,
    internet_service,
    COUNT(*)                                AS top_quartile_customers,
    SUM(churn)                              AS churned,
    ROUND(100.0 * AVG(churn), 2)            AS churn_rate_pct,
    ROUND(SUM(churn * monthly_charges) * 12, 0) AS annualised_revenue_lost
FROM ranked
WHERE spend_quartile = 4
GROUP BY contract_type, internet_service
ORDER BY annualised_revenue_lost DESC;
