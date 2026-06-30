"""Train a production XGBoost pipeline on the 20 live prediction features."""
import json
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


DATASET_CSV = "output/final_benter_dataset.csv"
OLD_XGB_MODEL = "models/benter_baseline_xgb.pkl"
NEW_XGB_MODEL = "models/xgboost_production.pkl"
COMPAT_REPORT = "reports/xgboost_feature_compatibility_report.md"
RETRAIN_REPORT = "reports/xgboost_retrain_report.md"
DEPRECATED_MARKER = "models/benter_baseline_xgb.deprecated.md"

FEATURE_COLS = [
    "track", "distance", "surface", "race_class", "carried_weight", "draw",
    "handicap_rating", "days_since_last_race", "last_3_avg_position",
    "last_5_avg_position", "last_10_avg_position", "surface_win_rate",
    "distance_win_rate", "track_win_rate", "jockey_horse_win_rate",
    "trainer_horse_win_rate", "weight_change", "class_change",
    "distance_change", "surface_change",
]

CATEGORICAL_COLS = ["track", "surface", "race_class"]
NUMERIC_COLS = [c for c in FEATURE_COLS if c not in CATEGORICAL_COLS]
LEAKAGE_CANDIDATES = [
    "finish_position", "finish_time_seconds", "race_time", "finish", "is_win",
    "winner", "result", "odds", "agf", "agf_percent", "agf_rank", "prize",
    "margin_text", "margin_lengths_numeric",
]


def normalize_by_race(frame, prob_col="probability"):
    out = frame.copy()
    out["norm_probability"] = 0.0
    for _, group in out.groupby("race_id"):
        idx = group.index
        total = group[prob_col].sum()
        if total > 0:
            out.loc[idx, "norm_probability"] = group[prob_col] / total
        elif len(group) > 0:
            out.loc[idx, "norm_probability"] = 1.0 / len(group)
    return out


def top1_accuracy(frame):
    winners = frame.sort_values("norm_probability", ascending=False).groupby("race_id").head(1)
    return float(winners["is_win"].mean()) if len(winners) else float("nan")


def legacy_main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    df = pd.read_csv(DATASET_CSV, low_memory=False)
    dataset_cols = list(df.columns)

    old_info = {}
    old_feature_names = []
    try:
        with open(OLD_XGB_MODEL, "rb") as f:
            old_model = pickle.load(f)
        old_info = {
            "type": type(old_model).__name__,
            "module": type(old_model).__module__,
            "n_features_in": int(getattr(old_model, "n_features_in_", -1)),
            "booster_num_features": int(old_model.get_booster().num_features()) if hasattr(old_model, "get_booster") else None,
            "has_feature_names": bool(getattr(old_model.get_booster(), "feature_names", None)) if hasattr(old_model, "get_booster") else False,
        }
        if hasattr(old_model, "get_booster") and old_model.get_booster().feature_names:
            old_feature_names = list(old_model.get_booster().feature_names)
    except Exception as exc:
        old_info = {"load_error": repr(exc)}

    try:
        with open("models/benter_baseline_logistic.pkl", "rb") as f:
            logistic = pickle.load(f)
        expected_222_names = list(logistic.named_steps["preprocessor"].get_feature_names_out())
    except Exception:
        expected_222_names = []

    missing_production = [c for c in FEATURE_COLS if c not in dataset_cols]
    leakage_intersections = [c for c in LEAKAGE_CANDIDATES if c in FEATURE_COLS]
    generated_222_base_cols = sorted({name.split("__", 1)[-1].split("_", 1)[0] for name in expected_222_names if "__" in name})

    with open(COMPAT_REPORT, "w", encoding="utf-8") as report:
        report.write("# XGBoost Feature Compatibility Report\n\n")
        report.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        report.write("## Existing Model\n\n")
        report.write(f"- File: `{OLD_XGB_MODEL}`\n")
        report.write(f"- Type: `{old_info.get('module', '')}.{old_info.get('type', '')}`\n")
        report.write(f"- `n_features_in_`: `{old_info.get('n_features_in')}`\n")
        report.write(f"- Booster feature count: `{old_info.get('booster_num_features')}`\n")
        report.write(f"- Booster feature names available: `{old_info.get('has_feature_names')}`\n\n")
        report.write("## 222 Feature Finding\n\n")
        report.write("The existing XGBoost pickle is a bare `XGBClassifier`, not a preprocessing pipeline. It expects the post-transform matrix with 222 columns. The model does not retain feature names in its booster, so the named 222-feature list was reconstructed from the saved Logistic Regression pipeline preprocessor, which uses the same 20 production input columns.\n\n")
        report.write(f"- Reconstructed transformed feature count: `{len(expected_222_names)}`\n")
        report.write("- First transformed features:\n\n")
        for name in expected_222_names[:40]:
            report.write(f"  - `{name}`\n")
        report.write("\n## Dataset Compatibility\n\n")
        report.write(f"- Final dataset columns: `{len(dataset_cols)}`\n")
        report.write(f"- Production feature columns requested: `{len(FEATURE_COLS)}`\n")
        report.write(f"- Missing production features: `{missing_production}`\n")
        report.write(f"- Leakage intersections in production features: `{leakage_intersections}`\n\n")
        report.write("## Decision\n\n")
        report.write("The old XGBoost model is marked deprecated because it cannot accept the live 20-column prediction dataframe directly. A new production XGBoost pipeline is trained below with its own preprocessor and saved as `models/xgboost_production.pkl`.\n")

    usable = df.copy()
    usable["race_date_parsed"] = pd.to_datetime(usable["race_date"], errors="coerce")
    usable["finish_position_numeric"] = pd.to_numeric(usable["finish_position"], errors="coerce")
    usable = usable[usable["race_date_parsed"].notna() & usable["finish_position_numeric"].notna()].copy()
    usable = usable[usable["race_id"].notna()].copy()
    usable["is_win"] = (usable["finish_position_numeric"] == 1).astype(int)
    usable = usable.sort_values(["race_date_parsed", "race_id", "horse_id"])

    unique_dates = sorted(usable["race_date_parsed"].dt.date.unique())
    cutoff_date = unique_dates[int(len(unique_dates) * 0.8)]
    train = usable[usable["race_date_parsed"].dt.date <= cutoff_date].copy()
    test = usable[usable["race_date_parsed"].dt.date > cutoff_date].copy()
    if test.empty:
        split_idx = int(len(usable) * 0.8)
        train = usable.iloc[:split_idx].copy()
        test = usable.iloc[split_idx:].copy()
        cutoff_date = train["race_date_parsed"].max().date()

    for col in CATEGORICAL_COLS:
        train[col] = train[col].astype("string").fillna("missing")
        test[col] = test[col].astype("string").fillna("missing")
    for col in NUMERIC_COLS:
        train[col] = pd.to_numeric(train[col], errors="coerce")
        test[col] = pd.to_numeric(test[col], errors="coerce")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_COLS),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), CATEGORICAL_COLS),
        ]
    )

    pos = max(1, int(train["is_win"].sum()))
    neg = max(1, int((train["is_win"] == 0).sum()))
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=220,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=4,
        scale_pos_weight=neg / pos,
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])

    pipeline.fit(train[FEATURE_COLS], train["is_win"])
    raw_prob = pipeline.predict_proba(test[FEATURE_COLS])[:, 1]
    eval_df = test[["race_id", "horse_id", "is_win"]].copy()
    eval_df["probability"] = raw_prob
    eval_df = normalize_by_race(eval_df)

    raw_logloss = log_loss(test["is_win"], raw_prob, labels=[0, 1])
    raw_brier = brier_score_loss(test["is_win"], raw_prob)
    race_logloss = log_loss(eval_df["is_win"], eval_df["norm_probability"], labels=[0, 1])
    race_brier = brier_score_loss(eval_df["is_win"], eval_df["norm_probability"])
    acc = top1_accuracy(eval_df)

    with open(NEW_XGB_MODEL, "wb") as f:
        pickle.dump(pipeline, f)

    with open(DEPRECATED_MARKER, "w", encoding="utf-8") as f:
        f.write("# Deprecated XGBoost Model\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("`models/benter_baseline_xgb.pkl` is deprecated for production inference because it expects a 222-column transformed matrix and does not include the preprocessing pipeline. Use `models/xgboost_production.pkl` instead.\n")

    transformed_count = int(pipeline.named_steps["preprocessor"].transform(train[FEATURE_COLS].head(1)).shape[1])
    metrics = {
        "dataset_rows": int(len(df)),
        "usable_completed_rows": int(len(usable)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "cutoff_date": str(cutoff_date),
        "train_date_min": str(train["race_date_parsed"].min().date()),
        "train_date_max": str(train["race_date_parsed"].max().date()),
        "test_date_min": str(test["race_date_parsed"].min().date()),
        "test_date_max": str(test["race_date_parsed"].max().date()),
        "raw_logloss": float(raw_logloss),
        "raw_brier": float(raw_brier),
        "race_normalized_logloss": float(race_logloss),
        "race_normalized_brier": float(race_brier),
        "top1_accuracy": float(acc),
        "transformed_feature_count": transformed_count,
        "probability_min": float(raw_prob.min()),
        "probability_max": float(raw_prob.max()),
    }

    with open(RETRAIN_REPORT, "w", encoding="utf-8") as report:
        report.write("# XGBoost Retrain Report\n\n")
        report.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        report.write("## Output\n\n")
        report.write(f"- Production model: `{NEW_XGB_MODEL}`\n")
        report.write(f"- Deprecated marker: `{DEPRECATED_MARKER}`\n\n")
        report.write("## Features\n\n")
        report.write(f"- Production input features: `{len(FEATURE_COLS)}`\n")
        report.write(f"- Transformed feature count inside pipeline: `{transformed_count}`\n")
        report.write(f"- Categorical: `{CATEGORICAL_COLS}`\n")
        report.write(f"- Numeric: `{NUMERIC_COLS}`\n")
        report.write("- Leakage columns used: `[]`\n\n")
        report.write("## Time Split\n\n")
        report.write(f"- Train rows: `{metrics['train_rows']}`\n")
        report.write(f"- Test rows: `{metrics['test_rows']}`\n")
        report.write(f"- Cutoff date: `{metrics['cutoff_date']}`\n")
        report.write(f"- Train date range: `{metrics['train_date_min']}` to `{metrics['train_date_max']}`\n")
        report.write(f"- Test date range: `{metrics['test_date_min']}` to `{metrics['test_date_max']}`\n\n")
        report.write("## Metrics\n\n")
        report.write(f"- Raw log loss: `{metrics['raw_logloss']:.6f}`\n")
        report.write(f"- Raw Brier: `{metrics['raw_brier']:.6f}`\n")
        report.write(f"- Race-normalized log loss: `{metrics['race_normalized_logloss']:.6f}`\n")
        report.write(f"- Race-normalized Brier: `{metrics['race_normalized_brier']:.6f}`\n")
        report.write(f"- Top-1 accuracy: `{metrics['top1_accuracy']:.4%}`\n")
        report.write(f"- Probability min/max: `{metrics['probability_min']:.8f}` / `{metrics['probability_max']:.8f}`\n")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def main():
    """Compatibility entry point for the versioned three-model trainer."""
    from train_production_models import main as train_model_family
    return train_model_family()


if __name__ == "__main__":
    raise SystemExit(main())
