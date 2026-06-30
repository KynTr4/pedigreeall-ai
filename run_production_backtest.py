"""Leakage-safe, race-level temporal backtest for the production model family."""
from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    ndcg_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import DMatrix, XGBClassifier
from feature_contract import CATEGORICAL_FEATURES, MODEL_FEATURES, POST_RACE_COLUMNS

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "output" / "final_benter_dataset.parquet"
OUTPUT = ROOT / "output"
REPORTS = ROOT / "reports"

FEATURE_COLS = MODEL_FEATURES
CATEGORICAL_COLS = CATEGORICAL_FEATURES
NUMERIC_COLS = [c for c in FEATURE_COLS if c not in CATEGORICAL_COLS]
LEAKAGE_COLS = {
    "finish_position", "finish_time_seconds", "odds", "agf", "agf_percent",
    "agf_rank", "prize", "margin_text", "margin_lengths_numeric", "is_win",
    "handicap_rating", "result_handicap_rating", *POST_RACE_COLUMNS,
}


@dataclass(frozen=True)
class Fold:
    name: str
    year: int


FOLDS = [Fold("validation", 2024), Fold("test", 2025), Fold("holdout", 2026)]
MODEL_NAMES = ["logistic", "catboost", "xgboost"]


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[FEATURE_COLS].copy()
    for col in CATEGORICAL_COLS:
        out[col] = out[col].astype(object)
        out.loc[out[col].isna(), col] = "missing"
        out[col] = out[col].map(lambda value: "missing" if pd.isna(value) else str(value))
    for col in NUMERIC_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def build_models(y_train: pd.Series) -> dict[str, object]:
    logistic_preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), NUMERIC_COLS),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL_COLS),
    ])
    xgb_preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC_COLS),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL_COLS),
    ])
    positive = max(1, int(y_train.sum()))
    negative = max(1, int((y_train == 0).sum()))
    return {
        "logistic": Pipeline([
            ("preprocessor", logistic_preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "catboost": CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=False,
            thread_count=1,
            cat_features=CATEGORICAL_COLS,
            allow_writing_files=False,
        ),
        "xgboost": Pipeline([
            ("preprocessor", xgb_preprocessor),
            ("classifier", XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=220,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=1,
                scale_pos_weight=negative / positive,
            )),
        ]),
    }


def normalize_by_race(frame: pd.DataFrame, raw_col: str) -> pd.Series:
    totals = frame.groupby("race_id")[raw_col].transform("sum")
    counts = frame.groupby("race_id")[raw_col].transform("size")
    return np.where(totals > 0, frame[raw_col] / totals, 1.0 / counts)


def expected_calibration_error(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    assigned = np.clip(np.digitize(probability, edges, right=True) - 1, 0, bins - 1)
    ece = 0.0
    for index in range(bins):
        mask = assigned == index
        if mask.any():
            ece += mask.mean() * abs(float(y[mask].mean()) - float(probability[mask].mean()))
    return float(ece)


def race_rank_metrics(frame: pd.DataFrame, probability_col: str) -> tuple[float, float, float]:
    correlations: list[float] = []
    ndcgs: list[float] = []
    reciprocal_ranks: list[float] = []
    for _, race in frame.groupby("race_id", sort=False):
        probability = race[probability_col].to_numpy(dtype=float)
        finish = race["finish_position_numeric"].to_numpy(dtype=float)
        corr = spearmanr(probability, -finish).statistic
        if not math.isnan(corr):
            correlations.append(float(corr))
        ndcgs.append(float(ndcg_score([race["is_win"].to_numpy()], [probability], k=len(race))))
        ordered = race.assign(_p=probability).sort_values("_p", ascending=False)
        winner_positions = np.flatnonzero(ordered["is_win"].to_numpy() == 1)
        if winner_positions.size:
            reciprocal_ranks.append(1.0 / (int(winner_positions[0]) + 1))
    return (
        float(np.mean(correlations)) if correlations else float("nan"),
        float(np.mean(ndcgs)) if ndcgs else float("nan"),
        float(np.mean(reciprocal_ranks)) if reciprocal_ranks else float("nan"),
    )


def score_model(frame: pd.DataFrame, model_name: str, split: str) -> dict[str, object]:
    probability_col = f"{model_name}_norm_prob"
    y = frame["is_win"].to_numpy(dtype=int)
    probability = frame[probability_col].to_numpy(dtype=float)
    predicted_binary = (probability >= 0.5).astype(int)
    ranks = frame.groupby("race_id")[probability_col].rank(method="first", ascending=False)
    race_count = int(frame["race_id"].nunique())
    rank_corr, ndcg, mrr = race_rank_metrics(frame, probability_col)
    tn, fp, fn, tp = confusion_matrix(y, predicted_binary, labels=[0, 1]).ravel()
    return {
        "split": split,
        "model": model_name,
        "rows": int(len(frame)),
        "races": race_count,
        "accuracy": float(accuracy_score(y, predicted_binary)),
        "precision": float(precision_score(y, predicted_binary, zero_division=0)),
        "recall": float(recall_score(y, predicted_binary, zero_division=0)),
        "f1": float(f1_score(y, predicted_binary, zero_division=0)),
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
        "top1_winner_accuracy": float(frame.loc[ranks == 1, "is_win"].sum() / race_count),
        "top3_hit_rate": float(frame.loc[ranks <= 3].groupby("race_id")["is_win"].max().mean()),
        "top5_hit_rate": float(frame.loc[ranks <= 5].groupby("race_id")["is_win"].max().mean()),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "calibration_error": expected_calibration_error(y, probability),
        "rank_correlation": rank_corr,
        "ndcg": ndcg,
        "mean_reciprocal_rank": mrr,
    }


def calibration_rows(frame: pd.DataFrame, split: str, model_name: str, bins: int = 10) -> list[dict[str, object]]:
    probability_col = f"{model_name}_norm_prob"
    probability = frame[probability_col].to_numpy(dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    assigned = np.clip(np.digitize(probability, edges, right=True) - 1, 0, bins - 1)
    rows = []
    for index in range(bins):
        mask = assigned == index
        rows.append({
            "split": split,
            "model": model_name,
            "bin": index + 1,
            "bin_lower": edges[index],
            "bin_upper": edges[index + 1],
            "count": int(mask.sum()),
            "mean_predicted_probability": float(probability[mask].mean()) if mask.any() else np.nan,
            "observed_win_rate": float(frame.loc[mask, "is_win"].mean()) if mask.any() else np.nan,
        })
    return rows


def roi_race_rows(frame: pd.DataFrame, split: str, model_name: str) -> list[dict[str, object]]:
    probability_col = f"{model_name}_norm_prob"
    rows: list[dict[str, object]] = []
    for strategy, top_n in [("top_1", 1), ("top_2", 2), ("top_3", 3)]:
        cumulative_profit = 0.0
        peak = 0.0
        for race_id, race in frame.sort_values(["race_date_parsed", "race_id"]).groupby("race_id", sort=False):
            selections = race.nlargest(top_n, probability_col)
            stake = float(len(selections))
            winning = selections[selections["is_win"] == 1]
            valid_winning_odds = pd.to_numeric(winning["odds"], errors="coerce").dropna()
            valid_winning_odds = valid_winning_odds[valid_winning_odds > 0]
            payout = float(valid_winning_odds.iloc[0]) if len(valid_winning_odds) else 0.0
            profit = payout - stake
            cumulative_profit += profit
            peak = max(peak, cumulative_profit)
            rows.append({
                "split": split,
                "model": model_name,
                "strategy": strategy,
                "status": "ok",
                "race_id": race_id,
                "race_date": race["race_date_parsed"].iloc[0].date().isoformat(),
                "bets": int(stake),
                "winning_bets": int(len(winning) > 0 and len(valid_winning_odds) > 0),
                "stake": stake,
                "payout": payout,
                "profit": profit,
                "cumulative_profit": cumulative_profit,
                "drawdown": peak - cumulative_profit,
                "selected_average_odds": float(pd.to_numeric(selections["odds"], errors="coerce").mean()),
            })
    rows.append({
        "split": split,
        "model": model_name,
        "strategy": "value_bet_agf",
        "status": "unavailable_missing_agf",
        "race_id": "",
        "race_date": "",
        "bets": 0,
        "winning_bets": 0,
        "stake": 0.0,
        "payout": 0.0,
        "profit": 0.0,
        "cumulative_profit": 0.0,
        "drawdown": np.nan,
        "selected_average_odds": np.nan,
    })
    return rows


def raw_feature_from_transformed(name: str) -> str:
    suffix = name.split("__", 1)[-1]
    if suffix in NUMERIC_COLS:
        return suffix
    for col in CATEGORICAL_COLS:
        if suffix == col or suffix.startswith(f"{col}_"):
            return col
    return suffix


def aggregate_importance(names: list[str], values: np.ndarray) -> dict[str, float]:
    result = {feature: 0.0 for feature in FEATURE_COLS}
    for name, value in zip(names, values):
        raw = raw_feature_from_transformed(str(name))
        if raw in result:
            result[raw] += float(abs(value))
    return result


def feature_importance_rows(
    models: dict[str, object], X_train: pd.DataFrame, X_eval: pd.DataFrame, y_eval: pd.Series
) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    notes: list[str] = []
    sample = X_eval.sample(min(1000, len(X_eval)), random_state=42)
    permutation_sample = X_eval.sample(min(3000, len(X_eval)), random_state=43)
    permutation_y = y_eval.loc[permutation_sample.index]

    for model_name, model in models.items():
        try:
            perm = permutation_importance(
                model,
                permutation_sample,
                permutation_y,
                scoring="neg_log_loss",
                n_repeats=3,
                random_state=42,
                n_jobs=1,
            )
            permutation_values = dict(zip(FEATURE_COLS, perm.importances_mean))
        except Exception as exc:
            permutation_values = {feature: np.nan for feature in FEATURE_COLS}
            notes.append(f"{model_name} permutation importance failed: {exc!r}")

        shap_values = {feature: np.nan for feature in FEATURE_COLS}
        gain_values = {feature: np.nan for feature in FEATURE_COLS}
        try:
            if model_name == "catboost":
                pool = Pool(sample, cat_features=CATEGORICAL_COLS)
                native_shap = model.get_feature_importance(pool, type="ShapValues")[:, :-1]
                shap_values = dict(zip(FEATURE_COLS, np.mean(np.abs(native_shap), axis=0)))
                native_gain = model.get_feature_importance(type="PredictionValuesChange")
                gain_values = dict(zip(FEATURE_COLS, native_gain))
            else:
                preprocessor = model.named_steps["preprocessor"]
                classifier = model.named_steps["classifier"]
                transformed = preprocessor.transform(sample)
                names = list(preprocessor.get_feature_names_out())
                if model_name == "logistic":
                    dense = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
                    background = preprocessor.transform(X_train.sample(min(500, len(X_train)), random_state=41))
                    background = background.toarray() if hasattr(background, "toarray") else np.asarray(background)
                    try:
                        import shap

                        explainer = shap.LinearExplainer(classifier, background)
                        native_shap = np.asarray(explainer.shap_values(dense))
                    except Exception:
                        native_shap = (dense - background.mean(axis=0)) * classifier.coef_[0]
                        notes.append("Logistic SHAP used the exact centered linear contribution fallback.")
                    shap_values = aggregate_importance(names, np.mean(np.abs(native_shap), axis=0))
                    gain_values = {feature: np.nan for feature in FEATURE_COLS}
                else:
                    native_shap = classifier.get_booster().predict(
                        DMatrix(transformed), pred_contribs=True
                    )[:, :-1]
                    shap_values = aggregate_importance(names, np.mean(np.abs(native_shap), axis=0))
                    booster_gain = classifier.get_booster().get_score(importance_type="gain")
                    indexed = np.zeros(len(names), dtype=float)
                    for key, value in booster_gain.items():
                        if key.startswith("f") and key[1:].isdigit() and int(key[1:]) < len(indexed):
                            indexed[int(key[1:])] = value
                    gain_values = aggregate_importance(names, indexed)
        except Exception as exc:
            notes.append(f"{model_name} SHAP/gain failed: {exc!r}")

        for feature in FEATURE_COLS:
            rows.append({
                "model": model_name,
                "feature": feature,
                "shap_mean_abs": shap_values.get(feature, np.nan),
                "gain_importance": gain_values.get(feature, np.nan),
                "permutation_importance": permutation_values.get(feature, np.nan),
            })
    return rows, notes


def fmt(value: object, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{float(value):.{digits}f}"


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, divider]
    for _, row in frame[columns].iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def main() -> int:
    OUTPUT.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.read_parquet(DATASET)
    source_rows = int(len(df))
    source_columns = int(len(df.columns))
    source_duplicates = int(df.duplicated(["race_id", "horse_id"]).sum())
    missing_features = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing production features: {missing_features}")
    leakage_intersection = sorted(set(FEATURE_COLS) & LEAKAGE_COLS)
    if leakage_intersection:
        raise ValueError(f"Leakage features in production list: {leakage_intersection}")

    df["race_date_parsed"] = pd.to_datetime(
        df["race_date"], dayfirst=True, errors="coerce"
    )
    as_of_date = pd.Timestamp(date.today())
    future_dated_rows = int((df["race_date_parsed"] > as_of_date).sum())
    df = df[df["race_date_parsed"] <= as_of_date].copy()
    df["finish_position_numeric"] = pd.to_numeric(df["finish_position"], errors="coerce")
    df["odds"] = pd.to_numeric(df["odds"], errors="coerce")
    df = df[df["race_date_parsed"].notna() & df["finish_position_numeric"].notna()].copy()
    df["is_win"] = (df["finish_position_numeric"] == 1).astype(int)
    df["year"] = df["race_date_parsed"].dt.year
    if "race_field_complete" not in df.columns:
        raise ValueError("Dataset is missing fail-closed race_field_complete audit")
    complete = df["race_field_complete"].fillna(False).astype(bool)
    incomplete_races = int(df.loc[~complete, "race_id"].nunique())
    df = df[complete].copy()
    race_quality = df.groupby("race_id").agg(winners=("is_win", "sum"), runners=("horse_id", "size"))
    valid_races = race_quality[(race_quality["winners"] == 1) & (race_quality["runners"] >= 2)].index
    excluded_races = int(df["race_id"].nunique() - len(valid_races))
    df = df[df["race_id"].isin(valid_races)].sort_values(["race_date_parsed", "race_id", "horse_id"]).copy()

    all_predictions: list[pd.DataFrame] = []
    all_scores: list[dict[str, object]] = []
    all_calibration: list[dict[str, object]] = []
    all_roi: list[dict[str, object]] = []
    fold_audit: list[dict[str, object]] = []
    holdout_models: dict[str, object] = {}
    holdout_train_X = pd.DataFrame()
    holdout_eval_X = pd.DataFrame()
    holdout_eval_y = pd.Series(dtype=int)

    for fold in FOLDS:
        train = df[df["year"] < fold.year].copy()
        evaluation = df[df["year"] == fold.year].copy()
        if train.empty or evaluation.empty:
            raise ValueError(f"Fold {fold.name}/{fold.year} has empty train or evaluation data")
        if train["race_date_parsed"].max() >= evaluation["race_date_parsed"].min():
            raise AssertionError(f"Temporal leakage detected in {fold.name}")
        X_train = prepare_features(train)
        X_eval = prepare_features(evaluation)
        y_train = train["is_win"]
        models = build_models(y_train)
        fold_prediction = evaluation[[
            "race_id", "race_date", "race_date_parsed", "horse_id", "horse_name",
            "finish_position_numeric", "is_win", "odds", "agf_percent", "agf_rank",
            "had_jockey_change", "surface_change", "distance_change",
            "had_steward_incident", "incident_count_last_30d",
        ]].copy()
        fold_prediction["split"] = fold.name
        fold_prediction["evaluation_year"] = fold.year
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            raw = model.predict_proba(X_eval)[:, 1]
            if not np.isfinite(raw).all() or ((raw < 0) | (raw > 1)).any():
                raise ValueError(f"Invalid {model_name} probability in {fold.name}")
            fold_prediction[f"{model_name}_raw_prob"] = raw
            fold_prediction[f"{model_name}_norm_prob"] = normalize_by_race(
                fold_prediction, f"{model_name}_raw_prob"
            )
        fold_prediction["ensemble_raw_prob"] = fold_prediction[
            [f"{name}_raw_prob" for name in MODEL_NAMES]
        ].mean(axis=1)
        fold_prediction["ensemble_norm_prob"] = fold_prediction[
            [f"{name}_norm_prob" for name in MODEL_NAMES]
        ].mean(axis=1)

        winner_ids = fold_prediction.loc[fold_prediction["is_win"] == 1].set_index("race_id")["horse_id"]
        fold_prediction["actual_winner_horse_id"] = fold_prediction["race_id"].map(winner_ids)
        for model_name in MODEL_NAMES + ["ensemble"]:
            col = f"{model_name}_norm_prob"
            fold_prediction[f"{model_name}_rank"] = fold_prediction.groupby("race_id")[col].rank(
                method="first", ascending=False
            ).astype(int)
            predicted_ids = fold_prediction.loc[fold_prediction[f"{model_name}_rank"] == 1].set_index("race_id")["horse_id"]
            fold_prediction[f"{model_name}_predicted_winner_horse_id"] = fold_prediction["race_id"].map(predicted_ids)
            all_scores.append(score_model(fold_prediction, model_name, fold.name))
            all_calibration.extend(calibration_rows(fold_prediction, fold.name, model_name))
            all_roi.extend(roi_race_rows(fold_prediction, fold.name, model_name))

        fold_audit.append({
            "split": fold.name,
            "evaluation_year": fold.year,
            "train_rows": len(train),
            "train_races": train["race_id"].nunique(),
            "train_max_date": train["race_date_parsed"].max().date().isoformat(),
            "evaluation_rows": len(evaluation),
            "evaluation_races": evaluation["race_id"].nunique(),
            "evaluation_min_date": evaluation["race_date_parsed"].min().date().isoformat(),
            "evaluation_max_date": evaluation["race_date_parsed"].max().date().isoformat(),
        })
        all_predictions.append(fold_prediction)
        if fold.name == "holdout":
            holdout_models = models
            holdout_train_X = X_train
            holdout_eval_X = X_eval
            holdout_eval_y = evaluation["is_win"]
        print(f"completed fold={fold.name} year={fold.year} train={len(train)} eval={len(evaluation)}")

    predictions = pd.concat(all_predictions, ignore_index=True)
    for model_name in MODEL_NAMES + ["ensemble"]:
        all_scores.append(score_model(predictions, model_name, "all_evaluation_folds"))
        all_calibration.extend(calibration_rows(predictions, "all_evaluation_folds", model_name))
    scores = pd.DataFrame(all_scores)
    calibration = pd.DataFrame(all_calibration)
    roi = pd.DataFrame(all_roi)

    importance_rows, importance_notes = feature_importance_rows(
        holdout_models, holdout_train_X, holdout_eval_X, holdout_eval_y
    )
    importance = pd.DataFrame(importance_rows)
    for metric in ["shap_mean_abs", "gain_importance", "permutation_importance"]:
        importance[f"{metric}_rank"] = importance.groupby("model")[metric].rank(
            method="min", ascending=False, na_option="bottom"
        )

    prediction_output = predictions.drop(columns=["race_date_parsed"])
    prediction_output.to_csv(OUTPUT / "backtest_predictions_v2.csv", index=False, encoding="utf-8")
    scores.to_csv(OUTPUT / "model_scores_v2.csv", index=False, encoding="utf-8")
    roi.to_csv(OUTPUT / "roi_simulation_v2.csv", index=False, encoding="utf-8")
    calibration.to_csv(OUTPUT / "calibration_table_v2.csv", index=False, encoding="utf-8")

    plt.figure(figsize=(8, 7))
    aggregate_calibration = calibration[calibration["split"] == "all_evaluation_folds"]
    for model_name in MODEL_NAMES + ["ensemble"]:
        curve = aggregate_calibration[(aggregate_calibration["model"] == model_name) & (aggregate_calibration["count"] > 0)]
        plt.plot(curve["mean_predicted_probability"], curve["observed_win_rate"], marker="o", label=model_name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, label="perfect")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed win rate")
    plt.title("Race-normalized reliability curve")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(REPORTS / "calibration_curve_v2.png", dpi=150)
    plt.close()

    holdout_scores = scores[scores["split"] == "holdout"].sort_values(
        ["top1_winner_accuracy", "log_loss"], ascending=[False, True]
    )
    winner = str(holdout_scores.iloc[0]["model"])
    winner_score = holdout_scores.iloc[0]
    roi_ok = roi[roi["status"] == "ok"].copy()
    roi_summary = roi_ok.groupby(["split", "model", "strategy"], as_index=False).agg(
        total_bets=("stake", "sum"),
        winning_bets=("winning_bets", "sum"),
        payout=("payout", "sum"),
        profit=("profit", "sum"),
        average_odds=("selected_average_odds", "mean"),
        max_drawdown=("drawdown", "max"),
    )
    roi_summary["roi"] = roi_summary["profit"] / roi_summary["total_bets"]

    error_rows = []
    for model_name in MODEL_NAMES + ["ensemble"]:
        rank_col = f"{model_name}_rank"
        predicted = predictions[predictions[rank_col] == 1].copy()
        lost = predicted[predicted["is_win"] == 0]
        winning_rows = predictions[predictions["is_win"] == 1]
        favorite_available = pd.to_numeric(predictions["agf_rank"], errors="coerce").notna().any()
        error_rows.append({
            "model": model_name,
            "lost_races": len(lost),
            "agf_favorite_analysis": "available" if favorite_available else "unavailable",
            "predicted_horse_jockey_change_rate": pd.to_numeric(lost["had_jockey_change"], errors="coerce").mean(),
            "predicted_horse_surface_change_rate": pd.to_numeric(lost["surface_change"], errors="coerce").mean(),
            "predicted_horse_distance_change_rate": (pd.to_numeric(lost["distance_change"], errors="coerce").fillna(0) != 0).mean(),
            "predicted_horse_steward_incident_rate": pd.to_numeric(lost["had_steward_incident"], errors="coerce").mean(),
            "actual_winner_jockey_change_rate": pd.to_numeric(winning_rows["had_jockey_change"], errors="coerce").mean(),
            "actual_winner_surface_change_rate": pd.to_numeric(winning_rows["surface_change"], errors="coerce").mean(),
            "actual_winner_distance_change_rate": (pd.to_numeric(winning_rows["distance_change"], errors="coerce").fillna(0) != 0).mean(),
            "actual_winner_steward_incident_rate": pd.to_numeric(winning_rows["had_steward_incident"], errors="coerce").mean(),
        })
    errors = pd.DataFrame(error_rows)

    display_scores = scores.copy()
    for col in ["accuracy", "precision", "recall", "f1", "top1_winner_accuracy", "top3_hit_rate", "top5_hit_rate", "log_loss", "brier_score", "roc_auc", "pr_auc", "calibration_error", "rank_correlation", "ndcg", "mean_reciprocal_rank"]:
        display_scores[col] = display_scores[col].map(lambda value: fmt(value))

    with open(REPORTS / "model_comparison_v2.md", "w", encoding="utf-8") as report:
        report.write(f"# Model Comparison v2\n\nGenerated: {generated_at}\n\n")
        report.write("## Numeric Results\n\n")
        report.write(markdown_table(display_scores, [
            "split", "model", "races", "accuracy", "precision", "recall", "f1", "top1_winner_accuracy", "top3_hit_rate",
            "top5_hit_rate", "log_loss", "brier_score", "roc_auc", "pr_auc",
            "calibration_error", "rank_correlation", "ndcg", "mean_reciprocal_rank",
        ]))
        report.write("\n\n## Winner\n\n")
        report.write(f"Holdout-first selection chooses **{winner}**: Top-1 `{winner_score['top1_winner_accuracy']:.2%}`, Top-3 `{winner_score['top3_hit_rate']:.2%}`, log loss `{winner_score['log_loss']:.4f}`. Selection priority is holdout Top-1, then lower log loss.\n")

    with open(REPORTS / "calibration_report_v2.md", "w", encoding="utf-8") as report:
        report.write(f"# Calibration Report\n\nGenerated: {generated_at}\n\n")
        report.write("Probabilities are normalized independently inside every race before calibration measurement. ECE uses 10 fixed-width bins.\n\n")
        report.write(markdown_table(display_scores, ["split", "model", "calibration_error", "brier_score", "log_loss"]))
        report.write("\n\nCurve: `reports/calibration_curve_v2.png`; reliability data: `output/calibration_table_v2.csv`. Empty high-probability bins are retained with count zero.\n")

    display_roi = roi_summary.copy()
    for col in ["roi", "average_odds", "max_drawdown", "profit"]:
        display_roi[col] = display_roi[col].map(lambda value: fmt(value))
    with open(REPORTS / "roi_report_v2.md", "w", encoding="utf-8") as report:
        report.write(f"# ROI Simulation Report — NOT CERTIFIED\n\nGenerated: {generated_at}\n\n")
        report.write("Historical odds have no immutable pre-race timestamp. ROI is diagnostic only and must not be used as a live-return claim.\n\n")
        report.write("Each selected horse receives 1 unit. Decimal `odds` is treated as total return including stake. No commission, limit, slippage, dead heat, or late odds movement is modeled.\n\n")
        report.write(markdown_table(display_roi, [
            "split", "model", "strategy", "total_bets", "winning_bets", "profit", "roi", "average_odds", "max_drawdown",
        ]))
        report.write("\n\n## AGF Value Bet\n\nNot calculated. `agf` has zero populated rows and `agf_percent`/`agf_rank` contain only `not_found`; fabricating an AGF comparison would invalidate the test. The CSV records this strategy as `unavailable_missing_agf`.\n")

    with open(REPORTS / "feature_importance_v2.md", "w", encoding="utf-8") as report:
        report.write(f"# Feature Importance\n\nGenerated: {generated_at}\n\n")
        report.write("Importance is computed on the 2026 holdout model trained only on internally complete race fields before 2026. Historical current-race HP is forbidden; the contract uses one-race-lagged pre_race_handicap_rating.\n\n")
        for model_name in MODEL_NAMES:
            report.write(f"## {model_name.title()}\n\n")
            subset = importance[importance["model"] == model_name].sort_values("shap_mean_abs", ascending=False).head(50).copy()
            for col in ["shap_mean_abs", "gain_importance", "permutation_importance"]:
                subset[col] = subset[col].map(lambda value: fmt(value, 6))
            report.write(markdown_table(subset, ["feature", "shap_mean_abs", "gain_importance", "permutation_importance"]))
            report.write("\n\n")
        report.write("Logistic regression has no tree gain; its gain cells are intentionally N/A. CatBoost gain is native PredictionValuesChange; XGBoost gain is booster split gain.\n")
        if importance_notes:
            report.write("\n## Computation Notes\n\n" + "\n".join(f"- {note}" for note in importance_notes) + "\n")

    fold_frame = pd.DataFrame(fold_audit)
    error_display = errors.copy()
    for col in error_display.columns[3:]:
        error_display[col] = error_display[col].map(lambda value: fmt(value))
    top_features = importance.sort_values("shap_mean_abs", ascending=False).groupby("model").head(5)
    top_feature_text = "; ".join(
        f"{name}: {', '.join(group['feature'].tolist())}" for name, group in top_features.groupby("model")
    )
    holdout_winner_roi = roi_summary[
        (roi_summary["split"] == "holdout") & (roi_summary["model"] == winner) & (roi_summary["strategy"] == "top_1")
    ]
    expected_roi = float(holdout_winner_roi.iloc[0]["roi"]) if len(holdout_winner_roi) else float("nan")
    production_ready = False
    with open(REPORTS / "backtest_report_v2.md", "w", encoding="utf-8") as report:
        report.write(f"# Production Backtest Report v2\n\nGenerated: {generated_at}\n\n")
        report.write("## Executive Decision\n\n")
        report.write("- Production ready: **No, not yet**. The intended recent-year temporal test now runs successfully; betting-data quality and live validation gates remain.\n")
        report.write(f"- Best holdout model: **{winner}**.\n")
        report.write(f"- Expected winner accuracy from the 2026 holdout: **{winner_score['top1_winner_accuracy']:.2%}** across `{int(winner_score['races'])}` races.\n")
        report.write(f"- Observed top-1 ROI under stated assumptions: **{expected_roi:.2%}**. This is descriptive, not a guaranteed live return.\n")
        report.write(f"- Highest SHAP contributors: {top_feature_text}.\n\n")
        report.write("## Temporal Design\n\n")
        report.write(markdown_table(fold_frame, list(fold_frame.columns)))
        report.write("\n\nEvery fold was retrained from scratch with `train_date < evaluation_date`. Saved production model predictions were not reused. Validation, test and holdout evaluate 2024, 2025 and 2026 respectively.\n\n")
        report.write("## Data Integrity\n\n")
        report.write(f"- Source rows/columns: `{source_rows}` / `{source_columns}`.\n")
        report.write(f"- Backtest as-of date: `{as_of_date.date().isoformat()}`; future-dated rows excluded: `{future_dated_rows}`.\n")
        report.write(f"- Completed valid-race rows evaluated/trained: `{len(df)}`.\n")
        report.write(f"- Incomplete-field races excluded before training/evaluation: `{incomplete_races}`.\n")
        report.write(f"- Excluded races without exactly one winner or with fewer than two runners: `{excluded_races}`.\n")
        report.write(f"- Duplicate horse/race rows in source: `{source_duplicates}`.\n")
        report.write(f"- Leakage columns intersecting model features: `{leakage_intersection}`.\n")
        report.write("- AGF value-bet test remains unavailable because a reliable timestamped pre-race AGF snapshot is not present.\n\n")
        report.write("## Error Analysis\n\n")
        report.write(markdown_table(error_display, list(error_display.columns)))
        report.write("\n\nAGF-favorite loss analysis is unavailable. Commissioner, jockey, surface and distance indicators are reported as association rates only; they do not establish causality.\n\n")
        report.write("## Weaknesses And Final Work\n\n")
        report.write("- Preserve the DB-backed rebuild and rerun these recent-year splits after each material data refresh.\n")
        report.write("- Repair AGF ingestion before enabling value betting; preserve timestamped pre-race AGF snapshots.\n")
        report.write("- Confirm that historical odds are genuinely available pre-bet and encode dead heats, scratches, deductions, commissions and stake limits.\n")
        report.write("- Monitor live calibration and feature drift across the 2024/2025/2026 evaluation sequence.\n")
        report.write("- Use the selected model only after those gates pass; current results justify shadow mode, not unattended wagering.\n\n")
        report.write("## Artifacts\n\n")
        for artifact in [
            "output/backtest_predictions_v2.csv", "output/model_scores_v2.csv", "output/roi_simulation_v2.csv",
            "output/calibration_table_v2.csv", "reports/model_comparison_v2.md", "reports/calibration_report_v2.md",
            "reports/calibration_curve_v2.png", "reports/roi_report_v2.md", "reports/feature_importance_v2.md",
        ]:
            report.write(f"- `{artifact}`\n")

    summary = {
        "source_rows": source_rows,
        "as_of_date": as_of_date.date().isoformat(),
        "future_dated_rows_excluded": future_dated_rows,
        "backtest_prediction_rows": int(len(predictions)),
        "evaluation_races": int(predictions["race_id"].nunique()),
        "winner": winner,
        "holdout_top1": float(winner_score["top1_winner_accuracy"]),
        "holdout_top1_roi": expected_roi,
        "production_ready": production_ready,
        "importance_notes": importance_notes,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
