-- Does holding a given add-on go with lower churn? Only internet customers are
-- compared, otherwise "No internet service" drags every "No" group down.
SELECT
    b.service_name,
    ROUND(100.0 * AVG(CASE WHEN b.status = 'Yes' THEN v.churn END), 2) AS churn_with_pct,
    ROUND(100.0 * AVG(CASE WHEN b.status = 'No'  THEN v.churn END), 2) AS churn_without_pct,
    ROUND(100.0 * AVG(CASE WHEN b.status = 'No'  THEN v.churn END)
        - 100.0 * AVG(CASE WHEN b.status = 'Yes' THEN v.churn END), 2) AS churn_gap_pts,
    SUM(CASE WHEN b.status = 'Yes' THEN 1 ELSE 0 END)                   AS adopters,
    ROUND(100.0 * AVG(CASE WHEN b.status = 'Yes' THEN 1.0 ELSE 0 END), 1) AS adoption_pct
FROM bridge_customer_service b
JOIN vw_customer_360 v ON v.customer_id = b.customer_id
WHERE v.internet_service <> 'No'
GROUP BY b.service_name
ORDER BY churn_gap_pts DESC;
