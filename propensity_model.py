"""
n8n-style PLG -> Enterprise Propensity Model

Portfolio demo using SYNTHETIC DATA ONLY. No access to n8n internal data.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET = "converted_to_enterprise_90d"
NUMERIC_FEATURES = [
    "employee_count", "executions_30d", "execution_growth_30d",
    "active_builders_30d", "active_workflows_30d", "active_departments_30d",
    "production_workflow_pct", "days_active", "pricing_page_views_30d",
    "security_page_views_30d", "docs_enterprise_views_30d", "workflow_failure_rate",
]
BINARY_FEATURES = [
    "sso_interest", "git_environment_interest", "log_streaming_interest",
    "external_secrets_interest",
]
CATEGORICAL_FEATURES = ["current_plan", "industry", "region"]
MODEL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def make_synthetic_data(n_accounts=12000, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "account_id": [f"acct_{i:05d}" for i in range(n_accounts)],
        "employee_count": np.clip(np.exp(rng.normal(np.log(180), 1.25, n_accounts)).astype(int), 2, 50000),
        "current_plan": rng.choice(["community", "starter", "pro", "business"], n_accounts, p=[0.34,0.26,0.27,0.13]),
        "industry": rng.choice(["software","financial_services","consulting","retail","manufacturing","healthcare","other"], n_accounts, p=[0.28,0.14,0.12,0.10,0.10,0.08,0.18]),
        "region": rng.choice(["DACH","UKI","North America","Southern Europe","Nordics","Other"], n_accounts, p=[0.20,0.16,0.30,0.10,0.08,0.16]),
    })
    size_factor = np.log1p(df["employee_count"]) / np.log(50001)
    df["executions_30d"] = np.maximum(0, np.exp(rng.normal(8.2 + 3.0*size_factor, 1.35, n_accounts)).astype(int))
    df["execution_growth_30d"] = np.clip(rng.normal(0.18,0.55,n_accounts), -0.9, 4.0)
    df["active_builders_30d"] = np.maximum(1, rng.poisson(1.4 + 12*size_factor + np.log1p(df["executions_30d"])/5))
    df["active_workflows_30d"] = np.maximum(1, rng.poisson(2 + 2.2*df["active_builders_30d"]))
    df["active_departments_30d"] = np.clip(rng.poisson(0.6 + 0.22*df["active_builders_30d"]) + 1, 1, 12)
    df["production_workflow_pct"] = np.clip(rng.beta(3.2,2.2,n_accounts),0,1)
    df["days_active"] = rng.integers(7,1100,n_accounts)
    df["workflow_failure_rate"] = np.clip(rng.beta(1.5,25,n_accounts),0,0.5)
    need = 0.55*np.log1p(df["employee_count"]) + 0.35*np.log1p(df["active_builders_30d"]) + 0.30*np.log1p(df["active_departments_30d"]) + rng.normal(0,1.1,n_accounts)
    def b(offset):
        return rng.binomial(1, np.clip(sigmoid(need-offset),0.005,0.85))
    df["sso_interest"] = b(5.8)
    df["git_environment_interest"] = b(6.1)
    df["log_streaming_interest"] = b(6.6)
    df["external_secrets_interest"] = b(6.8)
    df["pricing_page_views_30d"] = rng.poisson(0.25 + 0.8*df["sso_interest"] + 0.35*np.maximum(df["execution_growth_30d"],0))
    df["security_page_views_30d"] = rng.poisson(0.15 + 1.25*df["sso_interest"] + 0.55*df["external_secrets_interest"])
    df["docs_enterprise_views_30d"] = rng.poisson(0.25 + 0.85*df["git_environment_interest"] + 0.75*df["log_streaming_interest"])
    handraiser_logit = -6.1 + 0.55*np.log1p(df["pricing_page_views_30d"]) + 0.45*df["sso_interest"] + 0.35*df["log_streaming_interest"]
    df["requested_sales_contact"] = rng.binomial(1, sigmoid(handraiser_logit))
    large = (df["employee_count"]>=500).astype(int)
    high_growth = (df["execution_growth_30d"]>=0.50).astype(int)
    multi_team = (df["active_departments_30d"]>=3).astype(int)
    governance = df["sso_interest"] + df["git_environment_interest"] + df["log_streaming_interest"] + df["external_secrets_interest"]
    logit = (-7.3 + 0.42*np.log1p(df["employee_count"]) + 0.28*np.log1p(df["executions_30d"]) + 0.80*np.maximum(df["execution_growth_30d"],0)
             + 0.16*df["active_builders_30d"] + 0.25*df["active_departments_30d"] + 0.65*df["production_workflow_pct"] + 0.46*governance
             + 0.16*df["pricing_page_views_30d"] + 0.11*df["security_page_views_30d"] - 1.25*df["workflow_failure_rate"]
             + 0.75*(large*high_growth) + 0.60*(multi_team*(governance>=1)) + 1.60*df["requested_sales_contact"] + rng.normal(0,0.55,n_accounts))
    df[TARGET] = rng.binomial(1, np.clip(sigmoid(logit),0.002,0.92))
    return df

def build_preprocessor():
    numeric = Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])
    categorical = Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("numeric",numeric,NUMERIC_FEATURES+BINARY_FEATURES),("categorical",categorical,CATEGORICAL_FEATURES)])

def make_models():
    return {
        "logistic_regression": Pipeline([("prep",build_preprocessor()),("model",LogisticRegression(max_iter=2000,class_weight="balanced",random_state=RANDOM_STATE))]),
        "gradient_boosting": Pipeline([("prep",build_preprocessor()),("model",HistGradientBoostingClassifier(learning_rate=0.06,max_iter=250,max_leaf_nodes=24,l2_regularization=0.5,random_state=RANDOM_STATE))]),
    }

def capacity_metrics(y_true, probability, top_pct):
    r = pd.DataFrame({"actual":np.asarray(y_true),"probability":probability}).sort_values("probability",ascending=False)
    k = max(1, int(np.ceil(len(r)*top_pct)))
    c = r.head(k)
    base = r["actual"].mean()
    precision_k = c["actual"].mean()
    recall_k = c["actual"].sum()/max(r["actual"].sum(),1)
    return {"sales_capacity_pct":top_pct,"accounts_contacted":k,"precision_at_k":precision_k,"recall_at_k":recall_k,"lift_at_k":precision_k/max(base,1e-9),"base_conversion_rate":base}

def evaluate(name, model, X_test, y_test, top_pct):
    p = model.predict_proba(X_test)[:,1]
    pred = (p>=0.5).astype(int)
    out = {"model":name,"roc_auc":roc_auc_score(y_test,p),"pr_auc":average_precision_score(y_test,p),"brier_score":brier_score_loss(y_test,p),"precision_at_0_5":precision_score(y_test,pred,zero_division=0),"recall_at_0_5":recall_score(y_test,pred,zero_division=0)}
    out.update(capacity_metrics(y_test,p,top_pct))
    return out

def reason_codes(row):
    reasons=[]
    if row["execution_growth_30d"]>=0.50: reasons.append("rapid execution growth")
    if row["active_builders_30d"]>=8: reasons.append("multi-builder adoption")
    if row["active_departments_30d"]>=3: reasons.append("cross-department adoption")
    if row["employee_count"]>=500: reasons.append("enterprise firmographic fit")
    if row["sso_interest"]: reasons.append("SSO interest")
    if row["git_environment_interest"]: reasons.append("Git/environments interest")
    if row["log_streaming_interest"]: reasons.append("log-streaming interest")
    if row["external_secrets_interest"]: reasons.append("external-secrets interest")
    if row["pricing_page_views_30d"]>=2: reasons.append("pricing intent")
    if row["production_workflow_pct"]>=0.70: reasons.append("production-heavy usage")
    return ", ".join(reasons[:4]) if reasons else "no dominant trigger"

def assign_actions(scored, top_pct):
    scored=scored.copy()
    scored["sales_action"]="Nurture / monitor"
    scored.loc[scored["requested_sales_contact"]==1,"sales_action"]="Direct to Sales"
    pool=scored["requested_sales_contact"]==0
    ranks=scored.loc[pool,"enterprise_propensity"].rank(pct=True,ascending=False,method="first")
    scored.loc[ranks.index[ranks<=top_pct],"sales_action"]="AE priority"
    scored.loc[ranks.index[(ranks>top_pct)&(ranks<=top_pct*3)],"sales_action"]="SDR research / outreach"
    return scored

def main(input_path=None, output_dir="outputs", top_pct=0.05):
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(input_path) if input_path else make_synthetic_data()
    if input_path is None: df.to_csv(output/"synthetic_n8n_accounts.csv",index=False)
    modelling=df[df["requested_sales_contact"]==0].copy()
    X=modelling[MODEL_FEATURES]; y=modelling[TARGET]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,stratify=y,random_state=RANDOM_STATE)
    models=make_models(); rows=[]
    for name,model in models.items():
        model.fit(X_train,y_train); rows.append(evaluate(name,model,X_test,y_test,top_pct))
    metrics=pd.DataFrame(rows).sort_values("pr_auc",ascending=False)
    metrics.to_csv(output/"model_metrics.csv",index=False)
    best_name=metrics.iloc[0]["model"]; best=models[best_name]; best.fit(X,y)
    joblib.dump(best,output/"enterprise_propensity_model.joblib")
    scored=df.copy(); scored["enterprise_propensity"]=best.predict_proba(df[MODEL_FEATURES])[:,1]
    scored=assign_actions(scored,top_pct); scored["why_now"]=scored.apply(reason_codes,axis=1)
    scored=scored.sort_values(["requested_sales_contact","enterprise_propensity"],ascending=[False,False])
    scored.to_csv(output/"scored_accounts.csv",index=False)
    imp=permutation_importance(best,X_test,y_test,scoring="average_precision",n_repeats=8,random_state=RANDOM_STATE)
    pd.DataFrame({"feature":MODEL_FEATURES,"importance_mean":imp.importances_mean,"importance_std":imp.importances_std}).sort_values("importance_mean",ascending=False).to_csv(output/"feature_importance.csv",index=False)
    print("\nMODEL COMPARISON")
    print(metrics.round(3).to_string(index=False))
    print(f"\nBest model: {best_name}")
    print("\nTOP SALES PRIORITIES")
    cols=["account_id","enterprise_propensity","employee_count","executions_30d","execution_growth_30d","active_builders_30d","active_departments_30d","sales_action","why_now"]
    print(scored[cols].head(20).to_string(index=False))

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",default=None)
    parser.add_argument("--output-dir",default="outputs")
    parser.add_argument("--sales-capacity-pct",type=float,default=0.05)
    args=parser.parse_args()
    main(args.input,args.output_dir,args.sales_capacity_pct)
