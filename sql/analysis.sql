-- Customer churn analysis queries (run against SQLite table `customers`)

-- name: overall_churn
SELECT COUNT(*) AS customers,
       SUM(churn) AS churned,
       ROUND(100.0 * AVG(churn), 2) AS churn_rate_pct,
       ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
FROM customers;

-- name: churn_by_contract
SELECT contract,
       COUNT(*) AS customers,
       ROUND(100.0 * AVG(churn), 2) AS churn_rate_pct,
       ROUND(SUM(CASE WHEN churn = 1 THEN monthly_charges ELSE 0 END), 0) AS monthly_revenue_lost
FROM customers
GROUP BY contract
ORDER BY churn_rate_pct DESC;

-- name: churn_by_tenure_cohort
SELECT CASE
         WHEN tenure <= 6  THEN '01: 0-6 months'
         WHEN tenure <= 12 THEN '02: 7-12 months'
         WHEN tenure <= 24 THEN '03: 13-24 months'
         WHEN tenure <= 48 THEN '04: 25-48 months'
         ELSE '05: 49+ months'
       END AS tenure_cohort,
       COUNT(*) AS customers,
       ROUND(100.0 * AVG(churn), 2) AS churn_rate_pct
FROM customers
GROUP BY tenure_cohort
ORDER BY tenure_cohort;

-- name: churn_by_segment
-- Segment = contract x internet service; keep segments with >= 100 customers
SELECT contract, internet_service,
       COUNT(*) AS customers,
       ROUND(100.0 * AVG(churn), 2) AS churn_rate_pct,
       ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
FROM customers
GROUP BY contract, internet_service
HAVING COUNT(*) >= 100
ORDER BY churn_rate_pct DESC;

-- name: churn_by_payment_method
SELECT payment_method,
       COUNT(*) AS customers,
       ROUND(100.0 * AVG(churn), 2) AS churn_rate_pct
FROM customers
GROUP BY payment_method
ORDER BY churn_rate_pct DESC;

-- name: revenue_at_risk
-- Monthly revenue from churned customers vs. total, and share by segment (window function)
SELECT contract,
       ROUND(SUM(CASE WHEN churn = 1 THEN monthly_charges ELSE 0 END), 0) AS churned_mrr,
       ROUND(100.0 * SUM(CASE WHEN churn = 1 THEN monthly_charges ELSE 0 END)
             / SUM(SUM(CASE WHEN churn = 1 THEN monthly_charges ELSE 0 END)) OVER (), 1) AS share_of_lost_mrr_pct
FROM customers
GROUP BY contract
ORDER BY churned_mrr DESC;

-- name: data_validation
-- Data-quality checks: duplicates, nulls, and TotalCharges consistency with tenure x MonthlyCharges
SELECT
  (SELECT COUNT(*) - COUNT(DISTINCT customer_id) FROM customers) AS duplicate_ids,
  (SELECT COUNT(*) FROM customers WHERE total_charges IS NULL) AS null_total_charges,
  (SELECT COUNT(*) FROM customers
     WHERE tenure > 0
       AND ABS(total_charges - tenure * monthly_charges) / (tenure * monthly_charges) > 0.25) AS charges_inconsistent_gt25pct;
