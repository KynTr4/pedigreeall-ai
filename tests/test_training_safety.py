from __future__ import annotations

import pandas as pd
import pytest

import build_final_dataset as builder
import train_production_models as trainer
from feature_contract import MODEL_FEATURES, validate_model_feature_contract


def test_current_postrace_rating_is_forbidden_by_contract():
    assert "pre_race_handicap_rating" in MODEL_FEATURES
    assert "handicap_rating" not in MODEL_FEATURES
    with pytest.raises(ValueError):
        validate_model_feature_contract([
            "handicap_rating" if name == "pre_race_handicap_rating" else name
            for name in MODEL_FEATURES
        ])


def test_historical_rating_is_lagged_and_same_day_is_quarantined():
    frame = pd.DataFrame([
        {"horse_id": "h1", "race_id": "r1", "_date": pd.Timestamp("2025-01-01"), "handicap_rating": 40.0},
        {"horse_id": "h1", "race_id": "r2", "_date": pd.Timestamp("2025-02-01"), "handicap_rating": 47.0},
        {"horse_id": "h1", "race_id": "r3", "_date": pd.Timestamp("2025-02-01"), "handicap_rating": 49.0},
        {"horse_id": "h2", "race_id": "r4", "_date": pd.Timestamp("2025-01-01"), "handicap_rating": 30.0},
        {"horse_id": "h2", "race_id": "r5", "_date": pd.Timestamp("2025-02-01"), "handicap_rating": 35.0},
    ])
    result = builder.add_historical_pre_race_rating(frame)
    assert pd.isna(result.loc[0, "pre_race_handicap_rating"])
    assert pd.isna(result.loc[1, "pre_race_handicap_rating"])
    assert pd.isna(result.loc[2, "pre_race_handicap_rating"])
    assert result.loc[4, "pre_race_handicap_rating"] == 30.0
    assert result["history_order_certified"].tolist() == [True, False, False, True, True]


def test_incomplete_race_field_is_marked_and_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "OUTPUT", tmp_path)
    frame = pd.DataFrame([
        {"race_id": "complete", "finish_position": 1},
        {"race_id": "complete", "finish_position": 2},
        {"race_id": "missing", "finish_position": 1},
        {"race_id": "missing", "finish_position": 3},
    ])
    result = builder.add_race_field_audit(frame)
    flags = result.groupby("race_id")["race_field_complete"].first().to_dict()
    assert flags == {"complete": True, "missing": False}
    missing = result[result["race_id"].eq("missing")].iloc[0]
    assert missing["found_starters"] == 2
    assert missing["expected_starters_min"] == 3
    assert missing["missing_starters_min"] == 1
    assert (tmp_path / "race_starter_coverage.csv").exists()


def test_production_split_parses_day_first_dates(tmp_path, monkeypatch):
    rows = []
    for race_id, race_date in [("train", "31.12.2025"), ("test", "01.02.2026")]:
        for finish in (1, 2):
            row = {
                "race_id": race_id, "race_date": race_date,
                "horse_id": f"{race_id}-{finish}", "finish_position": finish,
                "race_field_complete": True,
            }
            row.update({name: ("x" if name in {"track", "surface", "race_class"} else 1.0)
                        for name in MODEL_FEATURES})
            rows.append(row)
    path = tmp_path / "dataset.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    monkeypatch.setattr(trainer, "DATASET", path)
    train, test, audit = trainer.load_training_data()
    assert audit["train_max"] == "2025-12-31"
    assert audit["test_min"] == "2026-02-01"
    assert len(train) == len(test) == 2
