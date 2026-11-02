from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import train_ibm_benchmarks_with_point_in_time_typologies as pit
from encode_ibm_typology_features import load_transactions
from train_ibm_benchmarks import (
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
OUTPUT_DIR = (
    PROJECT_DIR / "outputs" / "ibm_aml" / "benchmark_4_typology_ablation"
)

# Each experiment adds only the stated point-in-time feature group to the
# transaction features. "all_typologies" adds the complete typology block.
FEATURE_GROUPS: dict[str, list[str]] = {
    "transaction_only": [],
    "fan": [
        "incoming_transaction_count",
        "unique_senders",
        "outgoing_transaction_count",
        "unique_recipients",
        "fan_in_flag",
        "fan_out_flag",
    ],
    "structuring": [
        "near_threshold_transaction_count",
        "maximum_near_threshold_daily_count",
        "structuring_event_count",
        "structuring_flag",
    ],
    "circular": [
        "two_account_cycle_count",
        "three_account_cycle_count",
        "two_way_circular_flag",
        "three_way_circular_flag",
        "circular_transfer_count",
        "circular_transfer_flag",
    ],
    "rapid_pass_through": [
        "rapid_pass_through_count",
        "average_rapid_pass_through_hours",
        "minimum_rapid_pass_through_hours",
        "rapid_pass_through_flag",
    ],
    "all_typologies": pit.BASE_TYPOLOGY_COLUMNS,
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run feature-group ablations with leakage-free point-in-time AML "
            "typology features. Model settings stay fixed across experiments."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument(
        "--rebuild-features",
        action="store_true",
        help="Recalculate the shared point-in-time feature cache.",
    )
    return parser.parse_args()


def selected_typology_columns(base_columns: list[str]) -> list[str]:
    return [
        f"{role}_{column}"
        for role in ("sender", "recipient")
        for column in base_columns
    ]


def attach_selected_typologies(
    transaction_features: pd.DataFrame,
    typology_block: np.ndarray,
    base_columns: list[str],
) -> pd.DataFrame:
    if not base_columns:
        return transaction_features.copy()

    column_positions = {
        name: position
        for position, name in enumerate(pit.TYPOLOGY_FEATURE_COLUMNS)
    }
    columns = selected_typology_columns(base_columns)
    positions = [column_positions[column] for column in columns]
    frame = pd.DataFrame(
        np.asarray(typology_block[:, positions]),
        index=transaction_features.index,
        columns=columns,
        dtype="float32",
    )
    return pd.concat([transaction_features, frame], axis=1)


def add_deltas(comparison: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in (
            "precision",
            "recall",
            "f1",
            "pr_auc",
            "false_positive_rate",
        )
        if column in comparison.columns
    ]
    rows: list[dict[str, object]] = []
    for model_name, model_rows in comparison.groupby("model"):
        baseline = model_rows.loc[
            model_rows["feature_set"] == "transaction_only"
        ].iloc[0]
        for _, row in model_rows.iterrows():
            result: dict[str, object] = {
                "model": model_name,
                "feature_set": row["feature_set"],
            }
            for metric in metric_columns:
                result[f"delta_{metric}"] = float(row[metric] - baseline[metric])
            rows.append(result)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_arguments()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading IBM transactions from: {args.input}")
    transactions = load_transactions(args.input)
    if not transactions["timestamp"].is_monotonic_increasing:
        transactions = transactions.sort_values(
            "timestamp", kind="stable"
        ).reset_index(drop=True)
    transactions = transactions.rename(columns={"is_laundering": "target"})

    train_end = int(len(transactions) * pit.TRAIN_FRACTION)
    validation_end = int(
        len(transactions)
        * (pit.TRAIN_FRACTION + pit.VALIDATION_FRACTION)
    )
    train, validation, test = pit.chronological_split(transactions)
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
        raise ValueError(
            "At least one chronological split has no laundering transactions."
        )

    typology_matrix = pit.load_or_build_point_in_time_features(
        transactions, args.rebuild_features
    )
    frequency_maps = create_frequency_maps(train)
    raw_train = create_features(train, frequency_maps)
    raw_validation = create_features(validation, frequency_maps)
    raw_test = create_features(test, frequency_maps)
    y_train = train["target"].astype("int8")
    y_validation = validation["target"].astype("int8")
    y_test = test["target"].astype("int8")

    # Determine the sampled training rows once. Every ablation therefore uses
    # exactly the same transactions and differs only in its feature columns.
    sampled_raw_train, y_train_sampled = sample_training_data(raw_train, y_train)
    sampled_indices = sampled_raw_train.index
    print(
        f"Sampled training rows: {len(sampled_indices):,} "
        f"({int(y_train_sampled.sum()):,} laundering)"
    )

    results: list[dict[str, object]] = []
    for feature_set, base_columns in FEATURE_GROUPS.items():
        print(f"\nPreparing ablation: {feature_set}")
        x_train_full = attach_selected_typologies(
            raw_train, typology_matrix[:train_end], base_columns
        )
        x_train = x_train_full.loc[sampled_indices]
        x_validation = attach_selected_typologies(
            raw_validation,
            typology_matrix[train_end:validation_end],
            base_columns,
        )
        x_test = attach_selected_typologies(
            raw_test, typology_matrix[validation_end:], base_columns
        )

        experiment_dir = OUTPUT_DIR / feature_set
        experiment_dir.mkdir(parents=True, exist_ok=True)
        for model_name, model in pit.create_models(y_train_sampled).items():
            print(f"Training {model_name}: {feature_set}")
            started = time.perf_counter()
            model.fit(x_train, y_train_sampled)
            training_seconds = time.perf_counter() - started

            validation_probabilities = model.predict_proba(x_validation)[:, 1]
            threshold = best_f1_threshold(
                y_validation.to_numpy(), validation_probabilities
            )
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
            metrics["feature_set"] = feature_set
            metrics["typology_feature_count"] = len(base_columns) * 2
            results.append(metrics)
            print(json.dumps(metrics, indent=2))

            (experiment_dir / f"{model_name}_metrics.json").write_text(
                json.dumps(metrics, indent=2), encoding="utf-8"
            )
            pit.save_feature_importance(
                model_name, model, list(x_train.columns), experiment_dir
            )
            if args.save_models:
                joblib.dump(
                    model,
                    experiment_dir / f"{model_name}.joblib",
                    compress=3,
                )

        del x_train_full, x_train, x_validation, x_test

    comparison = pd.DataFrame(results)
    comparison.to_csv(OUTPUT_DIR / "ablation_comparison.csv", index=False)
    deltas = add_deltas(comparison)
    deltas.to_csv(OUTPUT_DIR / "ablation_delta_vs_transaction_only.csv", index=False)
    print(f"\nAblation results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
