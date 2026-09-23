# PLG → Enterprise Propensity Model

Portfolio project for the question: **Which self-serve accounts should Sales contact when product adoption is much larger than seller capacity?**

**Synthetic data only. No n8n internal data. Not affiliated with n8n.**

## Signal families

1. Consumption & momentum: executions, execution growth, production usage.
2. Adoption breadth: active builders, workflows, departments, tenure.
3. Enterprise need / intent: SSO, Git/environments, log streaming, external secrets, pricing/security/enterprise-doc views.
4. Firmographic fit: employee count, industry, region, current plan.

Explicit hand-raisers are routed directly to Sales rather than being used to train the proactive propensity model.

## Models

- Logistic Regression: interpretable baseline.
- Histogram Gradient Boosting: captures nonlinear interactions.

The winner is selected using PR-AUC. The project also reports Precision@K, Recall@K and Lift@K because Sales capacity is constrained.

## Why propensity is only step one

Propensity predicts **who is likely to buy**. The stronger question is **who is more likely to buy because Sales contacted them**. With outreach-treatment data, the next production step is uplift / causal modelling (T-learner, X-learner or causal forest).

A final commercial score can become:

`incremental conversion uplift × expected ARR ÷ expected seller effort`

## Run

```bash
pip install -r requirements.txt
python propensity_model.py
```

Outputs include a synthetic dataset, model comparison, scored accounts, feature importance and a serialized model.

## Production changes

- Use a temporal rather than random train/test split.
- Build features in dbt/BigQuery from product, Salesforce and web-intent data.
- Prevent leakage by ensuring every feature existed before the prediction timestamp.
- Calibrate probabilities and monitor drift.
- Run controlled Sales-outreach experiments.
- Replace pure propensity ranking with uplift / incremental-revenue ranking.
- Write scores and reason codes back to Salesforce.
