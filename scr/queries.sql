-- RQ1: How does price elasticity vary across different products?
SELECT Category,
       COUNT(*)            AS n_transactions,
       AVG(price)           AS avg_price,
       AVG(purchase_count)  AS avg_demand,
       AVG(log_price)       AS avg_log_price,
       AVG(log_final_price) AS avg_log_final_price
FROM df
GROUP BY Category
ORDER BY avg_price DESC;


-- RQ2: Does dynamic pricing improve conversion rate and profit margin?
SELECT discount_bin,
       COUNT(*)            AS n_transactions,
       AVG(margin_pct)      AS avg_margin_pct,
       AVG(purchase_count)  AS avg_demand_proxy,
       AVG(final_price)     AS avg_final_price
FROM df
GROUP BY discount_bin
ORDER BY CASE discount_bin
    WHEN 'no_disc'  THEN 1
    WHEN 'light'    THEN 2
    WHEN 'moderate' THEN 3
    WHEN 'heavy'    THEN 4
    WHEN 'deep'     THEN 5
END;


-- RQ3: What is the optimal price point for maximizing revenue?
SELECT price_bin,
       COUNT(*)            AS n_transactions,
       AVG(final_price)     AS avg_final_price,
       SUM(final_price)     AS total_revenue,
       AVG(purchase_count)  AS avg_demand,
       AVG(margin_pct)      AS avg_margin_pct
FROM df
GROUP BY price_bin
ORDER BY CASE price_bin
    WHEN 'very_low'  THEN 1
    WHEN 'low'       THEN 2
    WHEN 'mid'       THEN 3
    WHEN 'high'      THEN 4
    WHEN 'very_high' THEN 5
END;