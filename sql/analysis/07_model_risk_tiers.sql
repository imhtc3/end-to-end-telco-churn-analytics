-- Populated after the scoring step. Compares predicted risk tiers to the
-- actual outcome for each scored model.
SELECT
    s.model_name,
    s.risk_tier,
    COUNT(*)                                  AS customers,
    ROUND(AVG(s.churn_probability), 3)        AS avg_predicted,
    ROUND(AVG(v.churn), 3)                    AS actual_rate,
    ROUND(SUM(v.monthly_charges), 2)          AS mrr_in_tier
FROM model_scores s
JOIN vw_customer_360 v ON v.customer_id = s.customer_id
GROUP BY s.model_name, s.risk_tier
ORDER BY s.model_name, avg_predicted DESC;
