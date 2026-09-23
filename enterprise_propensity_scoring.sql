WITH account_usage AS (
    SELECT
        account_id,
        SUM(
            CASE
                WHEN execution_date >= CURRENT_DATE() - 30
                THEN executions
                ELSE 0
            END
        ) AS executions_30d,
        SUM(
            CASE
                WHEN execution_date BETWEEN
                     CURRENT_DATE() - 60
                     AND CURRENT_DATE() - 31
                THEN executions
                ELSE 0
            END
        ) AS executions_previous_30d,
        COUNT(DISTINCT user_id) AS active_builders,
        COUNT(DISTINCT workflow_id) AS active_workflows,
        COUNT(DISTINCT department) AS active_departments
    FROM product_usage
    WHERE execution_date >= CURRENT_DATE() - 90
    GROUP BY account_id
),
features AS (
    SELECT
        a.account_id,
        a.company_name,
        a.employee_count,
        a.current_plan,
        u.executions_30d,
        SAFE_DIVIDE(
            u.executions_30d - u.executions_previous_30d,
            NULLIF(u.executions_previous_30d, 0)
        ) AS execution_growth_rate,
        u.active_builders,
        u.active_workflows,
        u.active_departments,
        a.security_page_views_30d,
        a.pricing_page_views_30d,
        a.requested_sso,
        a.requested_sales_contact
    FROM accounts a
    JOIN account_usage u
        USING(account_id)
)
SELECT
    *,
    LEAST(100,
        -- Consumption: max 30
        LEAST(20, executions_30d / 10000.0)
        + LEAST(10, GREATEST(0, execution_growth_rate * 10))
        -- Adoption breadth: max 20
        + LEAST(10, active_builders)
        + LEAST(5, active_departments * 2)
        + LEAST(5, active_workflows / 5.0)
        -- Firmographic fit: max 15
        + CASE
            WHEN employee_count >= 1000 THEN 15
            WHEN employee_count >= 250 THEN 10
            WHEN employee_count >= 100 THEN 5
            ELSE 0
          END
        -- Enterprise intent: max 35
        + LEAST(10, security_page_views_30d * 2)
        + LEAST(5, pricing_page_views_30d)
        + CASE WHEN requested_sso THEN 10 ELSE 0 END
        + CASE WHEN requested_sales_contact THEN 10 ELSE 0 END
    ) AS enterprise_propensity_score
FROM features
ORDER BY enterprise_propensity_score DESC;
