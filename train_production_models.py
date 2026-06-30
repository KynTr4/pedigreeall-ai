"""Train one reproducible production model family on leakage-safe columns."""
from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, log_loss, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from feature_contract import CATEGORICAL_FEATURES, FEATURE_CONTRACT_VERSION, MODEL_FEATURES

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "output" / "final_benter_dataset.parquet"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MANIFEST = MODELS / "production_model_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame[MODEL_FEATURES].copy()
    for column in CATEGORICAL_FEATURES:
        output[column] = output[column].astype(object)
        output.loc[output[column].isna(), column] = "missing"
        output[column] = output[column].map(
            lambda value: "missing" if pd.isna(value) else str(value)
        )
    for column in set(MODEL_FEATURES) - set(CATEGORICAL_FEATURES):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def build_models(y: pd.Series) -> dict[str, object]:
    numeric = [column for column in MODEL_FEATURES if column not in CATEGORICAL_FEATURES]
    logistic_preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL_FEATURES),
    ])
    xgb_preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL_FEATURES),
    ])
    positive = max(1, int(y.sum()))
    negative = max(1, int((y == 0).sum()))
    return {
        "logistic": Pipeline([
            ("preprocessor", logistic_preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "catboost": CatBoostClassifier(
            iterations=300, learning_rate=0.05, depth=6, random_seed=42,
            verbose=False, thread_count=1, cat_features=CATEGORICAL_FEATURES,
            allow_writing_files=False,
        ),
        "xgboost": Pipeline([
            ("preprocessor", xgb_preprocessor),
            ("classifier", XGBClassifier(
                objective="binary:logistic", eval_metric="logloss", n_estimators=220,
                max_depth=5, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                random_state=42, n_jobs=1, scale_pos_weight=negative / positive,
            )),
        ]),
    }


def metrics(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, float | int]:
    scored = frame[["race_id", "is_win"]].copy()
    scored["probability"] = probability
    total = scored.groupby("race_id")["probability"].transform("sum")
    count = scored.groupby("race_id")["probability"].transform("size")
    scored["normalized"] = np.where(total.gt(0), scored["probability"] / total, 1.0 / count)
    top = scored.loc[scored.groupby("race_id")["normalized"].idxmax()]
    predicted = (scored["normalized"] >= 0.5).astype(int)
    return {
        "rows": int(len(scored)), "races": int(scored["race_id"].nunique()),
        "top1_accuracy": float(top["is_win"].mean()),
        "log_loss": float(log_loss(scored["is_win"], scored["normalized"], labels=[0, 1])),
        "brier": float(brier_score_loss(scored["is_win"], scored["normalized"])),
        "precision": float(precision_score(scored["is_win"], predicted, zero_division=0)),
        "recall": float(recall_score(scored["is_win"], predicted, zero_division=0)),
        "f1": float(f1_score(scored["is_win"], predicted, zero_division=0)),
    }


def normalize_by_race(frame: pd.DataFrame, probability: np.ndarray) -> np.ndarray:
    values = pd.Series(probability, index=frame.index, dtype=float)
    total = values.groupby(frame["race_id"]).transform("sum")
    count = values.groupby(frame["race_id"]).transform("size")
    return np.where(total.gt(0), values / total, 1.0 / count)


def load_training_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    frame = pd.read_parquet(DATASET)
    missing = sorted(set(MODEL_FEATURES + ["race_field_complete"]) - set(frame.columns))
    if missing:
        raise ValueError(f"Training dataset is missing required columns: {missing}")
    frame["race_date_parsed"] = pd.to_datetime(
        frame["race_date"], dayfirst=True, errors="coerce"
    )
    frame["finish_numeric"] = pd.to_numeric(frame["finish_position"], errors="coerce")
    frame = frame[
        frame["race_date_parsed"].notna()
        & frame["finish_numeric"].notna()
        & frame["race_field_complete"].fillna(False).astype(bool)
    ].copy()
    frame["is_win"] = frame["finish_numeric"].eq(1).astype(int)
    quality = frame.groupby("race_id").agg(winners=("is_win", "sum"), runners=("horse_id", "size"))
    valid = quality[(quality["winners"] == 1) & (quality["runners"] >= 2)].index
    frame = frame[frame["race_id"].isin(valid)].sort_values(
        ["race_date_parsed", "race_id", "horse_id"], kind="stable"
    ).copy()
    train = frame[frame["race_date_parsed"].dt.year < 2026].copy()
    test = frame[frame["race_date_parsed"].dt.year == 2026].copy()
    if train.empty or test.empty or train["race_date_parsed"].max() >= test["race_date_parsed"].min():
        raise AssertionError("Production temporal split is invalid")
    audit = {
        "train_rows": int(len(train)), "train_races": int(train["race_id"].nunique()),
        "train_min": train["race_date_parsed"].min().date().isoformat(),
        "train_max": train["race_date_parsed"].max().date().isoformat(),
        "test_rows": int(len(test)), "test_races": int(test["race_id"].nunique()),
        "test_min": test["race_date_parsed"].min().date().isoformat(),
        "test_max": test["race_date_parsed"].max().date().isoformat(),
    }
    return train, test, audit


def main() -> int:
    MODELS.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)
    train, test, audit = load_training_data()
    x_train, x_test = prepare(train), prepare(test)
    models = build_models(train["is_win"])
    probabilities: dict[str, np.ndarray] = {}
    normalized_probabilities: dict[str, np.ndarray] = {}
    results: dict[str, dict[str, float | int]] = {}
    paths = {
        "logistic": MODELS / "benter_baseline_logistic.pkl",
        "catboost": MODELS / "benter_baseline_catboost.pkl",
        "xgboost": MODELS / "xgboost_production.pkl",
    }
    for name, model in models.items():
        model.fit(x_train, train["is_win"])
        probability = np.asarray(model.predict_proba(x_test)[:, 1], dtype=float)
        if not np.isfinite(probability).all():
            raise ValueError(f"{name} produced non-finite probabilities")
        probabilities[name] = probability
        normalized_probabilities[name] = normalize_by_race(test, probability)
        results[name] = metrics(test, probability)
        with paths[name].open("wb") as stream:
            pickle.dump(model, stream)
    # Production shadow_mode averages race-normalized member probabilities.
    ensemble = np.mean(list(normalized_probabilities.values()), axis=0)
    results["ensemble"] = metrics(test, ensemble)
    generated = datetime.now(timezone.utc).isoformat()
    manifest = {
        "generated_at": generated,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "features": MODEL_FEATURES,
        "dataset": str(DATASET.relative_to(ROOT)),
        "dataset_sha256": sha256(DATASET),
        "split": audit,
        "metrics": results,
        "artifacts": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for name, path in paths.items()},
        "ensemble": {"method": "equal_mean", "members": list(paths)},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Production Model Training Report v2", "", f"Generated: {generated}", "",
        "- Historical current-race `GET:Tjk/Get.HP` is forbidden.",
        "- `pre_race_handicap_rating` is one-race-lagged in historical training and direct pre-race program HANDICAP in live scoring.",
        "- Only internally complete race fields are admitted.",
        "- Date parser: `pd.to_datetime(..., dayfirst=True, errors='coerce')`.", "",
        f"- Train: {audit['train_rows']:,} rows / {audit['train_races']:,} races ({audit['train_min']}–{audit['train_max']})",
        f"- Holdout: {audit['test_rows']:,} rows / {audit['test_races']:,} races ({audit['test_min']}–{audit['test_max']})", "",
        "| Model | Top-1 | LogLoss | Brier | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in results.items():
        lines.append(
            f"| {name} | {value['top1_accuracy']:.4%} | {value['log_loss']:.4f} | "
            f"{value['brier']:.4f} | {value['precision']:.4f} | {value['recall']:.4f} | {value['f1']:.4f} |"
        )
    (REPORTS / "production_model_training_v2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
