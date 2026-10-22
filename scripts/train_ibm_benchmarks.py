from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

try:
    from xgboost import XGBClassifier
except ImportError as error:
    raise ImportError(
        "XGBoost is not installed. Run: python -m pip install --user xgboost"
    ) from error


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
DEFAULT_INPUT = PROJECT_DIR / "IBM_PUBLIC_DATASET" / "HI-Small_Trans.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "ibm_aml" / "benchmarks"

RANDOM_STATE = 42
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
NEGATIVE_TO_POSITIVE_TRAIN_RATIO = 20

RAW_COLUMNS = [
    "Timestamp",
    "From Bank",
    "To Bank",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
]

CATEGORICAL_COLUMNS = [
    "from_bank",
    "to_bank",
    "receiving_currency",
    "payment_currency",
    "payment_format",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Random Forest and XGBoost IBM AML benchmarks."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Save fitted models. Model files can be large.",
    )
    return parser.parse_args()


def load_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"IBM transaction file not found: {file_path}")

    data = pd.read_csv(
        file_path,
        usecols=RAW_COLUMNS,
        dtype={
            "From Bank": "string",
            "To Bank": "string",
            "Receiving Currency": "string",
            "Payment Currency": "string",
            "Payment Format": "string",
            "Is Laundering": "int8",
        },
    ).rename(
        columns={
            "Timestamp": "timestamp",
            "From Bank": "from_bank",
            "To Bank": "to_bank",
            "Amount Received": "amount_received",
            "Receiving Currency": "receiving_currency",
            "Amount Paid": "amount_paid",
            "Payment Currency": "payment_currency",
            "Payment Format": "payment_format",
            "Is Laundering": "target",
        }
    )

    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    if not data["timestamp"].is_monotonic_increasing:
        print("Sorting transactions chronologically...")
        data = data.sort_values("timestamp", kind="stable").reset_index(drop=True)

    invalid = set(data["target"].dropna().unique()) - {0, 1}
    if invalid:
        raise ValueError(f"Invalid Is Laundering labels: {invalid}")
    return data


def chronological_split(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end = int(len(data) * TRAIN_FRACTION)
    validation_end = int(len(data) * (TRAIN_FRACTION + VALIDATION_FRACTION))
    train = data.iloc[:train_end].copy()
    validation = data.iloc[train_end:validation_end].copy()
    test = data.iloc[validation_end:].copy()
    return train, validation, test


def create_frequency_maps(train: pd.DataFrame) -> dict[str, dict[str, float]]:
    maps: dict[str, dict[str, float]] = {}
    for column in CATEGORICAL_COLUMNS:
        frequencies = train[column].fillna("<missing>").value_counts(normalize=True)
        maps[column] = frequencies.to_dict()
    return maps


def create_features(
    data: pd.DataFrame,
    frequency_maps: dict[str, dict[str, float]],
) -> pd.DataFrame:
    timestamp = data["timestamp"]
    amount_paid = pd.to_numeric(data["amount_paid"], errors="coerce").fillna(0).clip(lower=0)
    amount_received = pd.to_numeric(data["amount_received"], errors="coerce").fillna(0).clip(lower=0)

    features = pd.DataFrame(index=data.index)
    features["log_amount_paid"] = np.log1p(amount_paid)
    features["log_amount_received"] = np.log1p(amount_received)
    features["amount_ratio"] = amount_received / amount_paid.replace(0, np.nan)
    features["amount_ratio"] = features["amount_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 100)
    amount_denominator = pd.concat(
        [amount_paid, amount_received], axis=1
    ).max(axis=1).replace(0, np.nan)
    features["amount_difference_ratio"] = (
        (amount_received - amount_paid).abs() / amount_denominator
    ).fillna(0)
    features["same_bank"] = (data["from_bank"] == data["to_bank"]).astype("int8")
    features["same_currency"] = (
        data["payment_currency"] == data["receiving_currency"]
    ).astype("int8")

    hour = timestamp.dt.hour.astype("float32")
    weekday = timestamp.dt.dayofweek.astype("float32")
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    features["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    features["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    features["is_weekend"] = (weekday >= 5).astype("int8")

    for column, mapping in frequency_maps.items():
        features[f"{column}_frequency"] = (
            data[column].fillna("<missing>").map(mapping).fillna(0).astype("float32")
        )

    return features.astype("float32")


def sample_training_data(
    features: pd.DataFrame, target: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    positive_indices = target[target == 1].index
    negative_indices = target[target == 0].index
    if len(positive_indices) == 0:
        raise ValueError("The chronological training period has no laundering transactions.")

    maximum_negatives = len(positive_indices) * NEGATIVE_TO_POSITIVE_TRAIN_RATIO
    selected_negatives = pd.Series(negative_indices).sample(
        n=min(len(negative_indices), maximum_negatives),
        random_state=RANDOM_STATE,
        replace=False,
    ).to_numpy()
    selected_indices = np.concatenate([positive_indices.to_numpy(), selected_negatives])
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(selected_indices)
    return features.loc[selected_indices], target.loc[selected_indices]


def best_f1_threshold(target: np.ndarray, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(target, probabilities)
    if len(thresholds) == 0:
        return 0.5
    scores = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    return float(thresholds[int(np.nanargmax(scores))])


def evaluate(
    model_name: str,
    target: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    training_seconds: float,
    inference_seconds: float,
) -> dict[str, float | int | str]:
    prediction = (probabilities >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(target, prediction, labels=[0, 1]).ravel()
    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0
    return {
        "model": model_name,
        "threshold": threshold,
        "precision": precision_score(target, prediction, zero_division=0),
        "recall": recall_score(target, prediction, zero_division=0),
        "f1": f1_score(target, prediction, zero_division=0),
        "pr_auc": average_precision_score(target, probabilities),
        "false_positive_rate": false_positive_rate,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
    }


def save_feature_importance(
    model_name: str, model: object, feature_names: list[str]
) -> None:
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(
        OUTPUT_DIR / f"{model_name}_feature_importance.csv", index=False
    )


def split_summary(name: str, data: pd.DataFrame) -> dict[str, object]:
    positives = int(data["target"].sum())
    return {
        "split": name,
        "rows": len(data),
        "laundering_transactions": positives,
        "normal_transactions": len(data) - positives,
        "laundering_rate": positives / len(data) if len(data) else 0,
        "start_timestamp": str(data["timestamp"].min()),
        "end_timestamp": str(data["timestamp"].max()),
    }


def main() -> None:
    args = parse_arguments()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading transactions from: {args.input}")
    data = load_data(args.input)
    train, validation, test = chronological_split(data)
    summary = pd.DataFrame(
        [
            split_summary("train", train),
            split_summary("validation", validation),
            split_summary("test", test),
        ]
    )
    summary.to_csv(OUTPUT_DIR / "chronological_split_summary.csv", index=False)
    print(summary.to_string(index=False))
    del data

    if any(part["target"].sum() == 0 for part in (train, validation, test)):
        raise ValueError(
            "At least one chronological split contains no laundering transactions. "
            "Review the split boundaries before training."
        )

    frequency_maps = create_frequency_maps(train)
    print("Creating numerical transaction features...")
    x_train_full = create_features(train, frequency_maps)
    x_validation = create_features(validation, frequency_maps)
    x_test = create_features(test, frequency_maps)
    y_train = train["target"].astype("int8")
    y_validation = validation["target"].astype("int8")
    y_test = test["target"].astype("int8")
    del train, validation, test
    x_train, y_train_sampled = sample_training_data(x_train_full, y_train)
    del x_train_full

    print(
        f"Sampled training rows: {len(x_train):,} "
        f"({int(y_train_sampled.sum()):,} laundering)"
    )

    models = {
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
            scale_pos_weight=(y_train_sampled.eq(0).sum() / y_train_sampled.eq(1).sum()),
        ),
    }

    results: list[dict[str, object]] = []
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
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
    print(f"\nComparison saved to: {OUTPUT_DIR / 'benchmark_comparison.csv'}")


if __name__ == "__main__":
    main()
