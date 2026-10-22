from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
except ImportError as error:
    raise ImportError(
        "XGBoost is not installed. Run: python -m pip install --user xgboost"
    ) from error

from encode_ibm_typology_features import (
    add_basic_and_fan_features,
    add_cycle_features,
    add_rapid_features,
    add_structuring_features,
    build_account_index,
    load_transactions,
)
from train_ibm_benchmarks import (
    RANDOM_STATE,
    best_f1_threshold,
    create_features,
    create_frequency_maps,
    evaluate,
    sample_training_data,
    split_summary,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
DEFAULT_INPUT = PROJECT_DIR / "IBM_PUBLIC_DATASET" / "HI-Small_Trans.csv"
IBM_OUTPUT_DIR = PROJECT_DIR / "outputs" / "ibm_aml"
RAW_BENCHMARK_DIR = IBM_OUTPUT_DIR / "benchmarks"
OUTPUT_DIR = IBM_OUTPUT_DIR / "benchmark_2_typology_features"
SNAPSHOT_DIR = OUTPUT_DIR / "typology_feature_snapshots"
COMBINED_COMPARISON_FILE = IBM_OUTPUT_DIR / "benchmark_1_vs_2_comparison.csv"

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

TYPOLOGY_COLUMNS = [
    "incoming_transaction_count",
    "unique_senders",
    "total_incoming_amount",
    "average_incoming_amount",
    "maximum_incoming_amount",
    "outgoing_transaction_count",
    "unique_recipients",
    "total_outgoing_amount",
    "average_outgoing_amount",
    "maximum_outgoing_amount",
    "fan_in_flag",
    "fan_out_flag",
    "near_threshold_transaction_count",
    "maximum_near_threshold_daily_count",
    "structuring_event_count",
    "structuring_flag",
    "two_account_cycle_count",
    "three_account_cycle_count",
    "two_way_circular_flag",
    "three_way_circular_flag",
    "circular_transfer_count",
    "circular_transfer_flag",
    "rapid_pass_through_count",
    "average_rapid_pass_through_hours",
    "minimum_rapid_pass_through_hours",
    "rapid_pass_through_flag",
    "typology_flag_count",
]

LOG_TRANSFORM_COLUMNS = {
    "incoming_transaction_count",
    "unique_senders",
    "total_incoming_amount",
    "average_incoming_amount",
    "maximum_incoming_amount",
    "outgoing_transaction_count",
    "unique_recipients",
    "total_outgoing_amount",
    "average_outgoing_amount",
    "maximum_outgoing_amount",
    "near_threshold_transaction_count",
    "maximum_near_threshold_daily_count",
    "structuring_event_count",
    "two_account_cycle_count",
    "three_account_cycle_count",
    "circular_transfer_count",
    "rapid_pass_through_count",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train IBM AML Random Forest and XGBoost benchmarks with "
            "graph-derived AML typology features."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument(
        "--rebuild-snapshots",
        action="store_true",
        help="Recalculate typology feature snapshots even if cached files exist.",
    )
    return parser.parse_args()


def chronological_split(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end = int(len(data) * TRAIN_FRACTION)
    validation_end = int(len(data) * (TRAIN_FRACTION + VALIDATION_FRACTION))
    return (
        data.iloc[:train_end].copy(),
        data.iloc[train_end:validation_end].copy(),
        data.iloc[validation_end:].copy(),
    )


def build_snapshot(history: pd.DataFrame, output_file: Path) -> pd.DataFrame:
    print(f"Building typology features from {len(history):,} earlier transactions...")
    accounts = build_account_index(history)
    features = accounts.copy()
    features = add_basic_and_fan_features(features, history)
    features = add_structuring_features(features, history)
    features = add_cycle_features(features, history, accounts)
    features, rapid_events = add_rapid_features(features, history)
    del rapid_events

    numeric_columns = features.select_dtypes(include=["number"]).columns
    features[numeric_columns] = features[numeric_columns].fillna(0)
    flag_columns = [
        "structuring_flag",
        "fan_in_flag",
        "fan_out_flag",
        "two_way_circular_flag",
        "three_way_circular_flag",
        "rapid_pass_through_flag",
    ]
    features["typology_flag_count"] = features[flag_columns].sum(axis=1).astype("int8")
    features.to_csv(output_file, index=False)
    print(f"Historical feature snapshot saved to: {output_file}")
    return features


def load_or_build_snapshot(
    history: pd.DataFrame,
    output_file: Path,
    rebuild: bool,
) -> pd.DataFrame:
    if output_file.exists() and not rebuild:
        print(f"Loading cached typology features: {output_file}")
        return pd.read_csv(output_file)
    return build_snapshot(history, output_file)


def attach_typology_features(
    raw_features: pd.DataFrame,
    transactions: pd.DataFrame,
    snapshot: pd.DataFrame,
) -> pd.DataFrame:
    snapshot = snapshot.set_index("account_key")
    output = raw_features.copy()

    for role, account_column in (
        ("sender", "sender_id"),
        ("recipient", "recipient_id"),
    ):
        account_keys = transactions[account_column]
        for column in TYPOLOGY_COLUMNS:
            if column not in snapshot.columns:
                values = pd.Series(0, index=transactions.index, dtype="float32")
            else:
                values = account_keys.map(snapshot[column]).fillna(0)

            values = pd.to_numeric(values, errors="coerce").fillna(0).clip(lower=0)
            if column in LOG_TRANSFORM_COLUMNS:
                values = np.log1p(values)
            output[f"{role}_{column}"] = values.astype("float32")

    return output


def save_feature_importance(
    model_name: str, model: object, feature_names: list[str]
) -> None:
    importance = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv(OUTPUT_DIR / f"{model_name}_feature_importance.csv", index=False)


def create_models(y_train: pd.Series) -> dict[str, object]:
    scale = y_train.eq(0).sum() / y_train.eq(1).sum()
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=20,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            scale_pos_weight=scale,
        ),
    }


def combine_comparisons(typology_results: pd.DataFrame) -> None:
    raw_file = RAW_BENCHMARK_DIR / "benchmark_comparison.csv"
    typology_results = typology_results.copy()
    typology_results.insert(0, "feature_set", "transaction_plus_typology")

    if raw_file.exists():
        raw_results = pd.read_csv(raw_file)
        raw_results.insert(0, "feature_set", "transaction_only")
        combined = pd.concat([raw_results, typology_results], ignore_index=True)
        combined.to_csv(COMBINED_COMPARISON_FILE, index=False)
        print(
            "Combined comparison saved to: "
            f"{COMBINED_COMPARISON_FILE}"
        )
    else:
        print(
            "Raw benchmark comparison was not found. Run train_ibm_benchmarks.py "
            "to generate the combined comparison."
        )


def main() -> None:
    args = parse_arguments()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading IBM transactions from: {args.input}")
    transactions = load_transactions(args.input)
    if not transactions["timestamp"].is_monotonic_increasing:
        transactions = transactions.sort_values("timestamp", kind="stable").reset_index(drop=True)
    transactions = transactions.rename(columns={"is_laundering": "target"})

    train, validation, test = chronological_split(transactions)
    summary = pd.DataFrame(
        [
            split_summary("train", train),
            split_summary("validation", validation),
            split_summary("test", test),
        ]
    )
    summary.to_csv(OUTPUT_DIR / "chronological_split_summary.csv", index=False)
    print(summary.to_string(index=False))

    if any(part["target"].sum() == 0 for part in (train, validation, test)):
        raise ValueError("At least one chronological split has no laundering transactions.")

    train_history_file = SNAPSHOT_DIR / "train_history_account_features.csv"
    train_validation_history_file = (
        SNAPSHOT_DIR / "train_validation_history_account_features.csv"
    )
    train_snapshot = load_or_build_snapshot(
        train.rename(columns={"target": "is_laundering"}),
        train_history_file,
        args.rebuild_snapshots,
    )
    train_validation_history = pd.concat([train, validation], ignore_index=True)
    train_validation_snapshot = load_or_build_snapshot(
        train_validation_history.rename(columns={"target": "is_laundering"}),
        train_validation_history_file,
        args.rebuild_snapshots,
    )
    del train_validation_history, transactions

    frequency_maps = create_frequency_maps(train)
    print("Creating transaction and AML typology features...")
    x_train = attach_typology_features(
        create_features(train, frequency_maps), train, train_snapshot
    )
    x_validation = attach_typology_features(
        create_features(validation, frequency_maps), validation, train_snapshot
    )
    x_test = attach_typology_features(
        create_features(test, frequency_maps), test, train_validation_snapshot
    )
    y_train = train["target"].astype("int8")
    y_validation = validation["target"].astype("int8")
    y_test = test["target"].astype("int8")
    del train, validation, test, train_snapshot, train_validation_snapshot

    x_train, y_train_sampled = sample_training_data(x_train, y_train)
    print(
        f"Sampled training rows: {len(x_train):,} "
        f"({int(y_train_sampled.sum()):,} laundering)"
    )

    results: list[dict[str, object]] = []
    for model_name, model in create_models(y_train_sampled).items():
        print(f"\nTraining {model_name} with AML typology features...")
        started = time.perf_counter()
        model.fit(x_train, y_train_sampled)
        training_seconds = time.perf_counter() - started

        validation_probabilities = model.predict_proba(x_validation)[:, 1]
        threshold = best_f1_threshold(y_validation.to_numpy(), validation_probabilities)
        started = time.perf_counter()
        test_probabilities = model.predict_proba(x_test)[:, 1]
        inference_seconds = time.perf_counter() - started

        metrics = evaluate(
            model_name,
            y_test,
            test_probabilities,
            threshold,
            training_seconds,
            inference_seconds,
        )
        metrics["feature_set"] = "transaction_plus_typology"
        results.append(metrics)
        print(json.dumps(metrics, indent=2))
        (OUTPUT_DIR / f"{model_name}_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        save_feature_importance(model_name, model, list(x_train.columns))
        if args.save_models:
            joblib.dump(model, OUTPUT_DIR / f"{model_name}.joblib", compress=3)

    comparison = pd.DataFrame(results)
    comparison.to_csv(OUTPUT_DIR / "benchmark_comparison.csv", index=False)
    combine_comparisons(comparison.drop(columns=["feature_set"]))
    print(f"Typology benchmark saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
