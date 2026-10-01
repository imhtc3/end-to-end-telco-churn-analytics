-- Monthly recurring revenue lost to churn, and what share of the total each
-- segment accounts for. Running total shows how concentrated the loss is.
WITH seg AS (
    SELECT
        contract_type || ' / ' || payment_method AS segment,
        SUM(CASE WHEN churn = 1 THEN monthly_charges ELSE 0 END) AS mrr_lost,
        SUM(churn)                                               AS churned
    FROM vw_customer_360
    GROUP BY contract_type, payment_method
)
SELECT
    segment,
    churned,
    ROUND(mrr_lost, 2)                                                      AS mrr_lost,
    ROUND(100.0 * mrr_lost / SUM(mrr_lost) OVER (), 2)                      AS pct_of_total_loss,
    ROUND(100.0 * SUM(mrr_lost) OVER (ORDER BY mrr_lost DESC
                                      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
          / SUM(mrr_lost) OVER (), 2)                                       AS cumulative_pct,
    RANK() OVER (ORDER BY mrr_lost DESC)                                    AS loss_rank
FROM seg
ORDER BY loss_rank;
