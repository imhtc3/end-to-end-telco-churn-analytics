SELECT
    payment_method,
    CASE paperless_billing WHEN 1 THEN 'Paperless' ELSE 'Paper' END AS billing,
    COUNT(*)                         AS customers,
    ROUND(100.0 * AVG(churn), 2)     AS churn_rate_pct,
    ROUND(AVG(tenure), 1)            AS avg_tenure
FROM vw_customer_360
GROUP BY payment_method, paperless_billing
HAVING COUNT(*) >= 50
ORDER BY churn_rate_pct DESC;
