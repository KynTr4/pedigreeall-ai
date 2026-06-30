"""Fail-closed feature schema contract shared by builders, predictors and CI."""
from __future__ import annotations

MODEL_FEATURES = [
    "track", "distance", "surface", "race_class", "carried_weight", "draw",
    "pre_race_handicap_rating", "days_since_last_race", "last_3_avg_position",
    "last_5_avg_position", "last_10_avg_position", "surface_win_rate",
    "distance_win_rate", "track_win_rate", "jockey_horse_win_rate",
    "trainer_horse_win_rate", "weight_change", "class_change",
    "distance_change", "surface_change",
]
FEATURE_CONTRACT_VERSION = "asof-model-features-v2-prerace-rating"
CATEGORICAL_FEATURES = ["track", "surface", "race_class"]
POST_RACE_COLUMNS = {
    "finish", "finish_position", "finish_time", "finish_time_seconds",
    "prize", "margin", "margin_text", "margin_lengths_numeric", "result_odds",
    "result_status", "is_win", "target", "label", "winner",
    # Historical horse_races.rating is GET:Tjk/Get.HP fetched after the race.
    # It must never be admitted directly as a model input.
    "handicap_rating", "result_handicap_rating",
}
MARKET_COLUMNS = {"odds", "agf", "agf_percent", "agf_rank"}
DIRECT_PROGRAM_FEATURES = {
    "track", "distance", "surface", "race_class", "carried_weight", "draw",
    "pre_race_handicap_rating",
}


def validate_model_feature_contract(features: list[str] | tuple[str, ...]) -> None:
    values = list(features)
    duplicates = sorted({name for name in values if values.count(name) > 1})
    forbidden = sorted(set(values) & (POST_RACE_COLUMNS | MARKET_COLUMNS))
    missing = sorted(set(MODEL_FEATURES) - set(values))
    extra = sorted(set(values) - set(MODEL_FEATURES))
    if duplicates or forbidden or missing or extra:
        raise ValueError(
            f"Unsafe model feature contract: duplicates={duplicates}, "
            f"forbidden={forbidden}, missing={missing}, extra={extra}"
        )


validate_model_feature_contract(MODEL_FEATURES)
