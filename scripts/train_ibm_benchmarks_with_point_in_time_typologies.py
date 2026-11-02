from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid

try:
    from xgboost import XGBClassifier
except ImportError as error:
    raise ImportError(
        "XGBoost is not installed. Run: python -m pip install --user xgboost"
    ) from error

from encode_ibm_typology_features import (
    FAN_IN_MIN_SENDERS,
    FAN_OUT_MIN_RECIPIENTS,
    RAPID_HOURS,
    RAPID_MAX_RATIO,
    RAPID_MIN_RATIO,
    STRUCTURING_LOWER_BOUND,
    STRUCTURING_MIN_TRANSACTIONS,
    STRUCTURING_THRESHOLD,
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
OLD_TYPOLOGY_DIR = IBM_OUTPUT_DIR / "benchmark_2_typology_features"
OUTPUT_DIR = IBM_OUTPUT_DIR / "benchmark_3_point_in_time_typologies"
TUNING_OUTPUT_DIR = OUTPUT_DIR / "tuning"
POINT_IN_TIME_FILE = OUTPUT_DIR / "point_in_time_typology_features.npy"
POINT_IN_TIME_METADATA_FILE = OUTPUT_DIR / "point_in_time_typology_features.json"
COMBINED_COMPARISON_FILE = (
    IBM_OUTPUT_DIR / "transaction_only_vs_typology_comparison.csv"
)

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
FEATURE_CACHE_VERSION = 1

BASE_TYPOLOGY_COLUMNS = [
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

TYPOLOGY_FEATURE_COLUMNS = [
    f"{role}_{column}"
    for role in ("sender", "recipient")
    for column in BASE_TYPOLOGY_COLUMNS
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


@dataclass
class AccountState:
    incoming_count: int = 0
    incoming_total: float = 0.0
    incoming_maximum: float = 0.0
    incoming_senders: set[int] = field(default_factory=set)

    outgoing_count: int = 0
    outgoing_total: float = 0.0
    outgoing_maximum: float = 0.0
    outgoing_recipients: set[int] = field(default_factory=set)

    near_threshold_count: int = 0
    maximum_near_threshold_daily_count: int = 0
    structuring_event_count: int = 0

    two_account_cycle_count: int = 0
    three_account_cycle_count: int = 0

    rapid_count: int = 0
    rapid_delay_total: float = 0.0
    rapid_delay_minimum: float = float("inf")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Random Forest and XGBoost with leakage-free, point-in-time "
            "AML typology features."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument(
        "--rebuild-features",
        action="store_true",
        help="Recalculate the point-in-time feature cache.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help=(
            "Select model hyperparameters using validation PR-AUC. "
            "Tuning results are saved separately and do not overwrite the "
            "untuned benchmark."
        ),
    )
    parser.add_argument(
        "--tune-model",
        choices=("random_forest", "xgboost", "both"),
        default="both",
        help="Model to tune when --tune is enabled (default: both).",
    )
    return parser.parse_args()


#CHRONOLOGICALLY SPLIT DATA

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


def encode_accounts(
    transactions: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Convert sender and recipient account keys to compact integer node IDs."""
    both = pd.concat(
        [transactions["sender_id"], transactions["recipient_id"]],
        ignore_index=True,
    )
    codes, unique_accounts = pd.factorize(both, sort=False)
    row_count = len(transactions)
    sender_nodes = codes[:row_count].astype(np.int32, copy=False)
    recipient_nodes = codes[row_count:].astype(np.int32, copy=False)
    return sender_nodes, recipient_nodes, len(unique_accounts)


def state_values(state: AccountState | None) -> np.ndarray:
    """Return the 27 historical features for one account."""
    if state is None:
        return np.zeros(len(BASE_TYPOLOGY_COLUMNS), dtype=np.float32)

    unique_senders = len(state.incoming_senders)
    unique_recipients = len(state.outgoing_recipients)
    average_incoming = (
        state.incoming_total / state.incoming_count if state.incoming_count else 0.0
    )
    average_outgoing = (
        state.outgoing_total / state.outgoing_count if state.outgoing_count else 0.0
    )
    average_rapid_delay = (
        state.rapid_delay_total / state.rapid_count if state.rapid_count else 0.0
    )
    minimum_rapid_delay = (
        state.rapid_delay_minimum if state.rapid_count else 0.0
    )

    fan_in_flag = int(unique_senders >= FAN_IN_MIN_SENDERS)
    fan_out_flag = int(unique_recipients >= FAN_OUT_MIN_RECIPIENTS)
    structuring_flag = int(state.structuring_event_count > 0)
    two_way_flag = int(state.two_account_cycle_count > 0)
    three_way_flag = int(state.three_account_cycle_count > 0)
    circular_count = (
        state.two_account_cycle_count + state.three_account_cycle_count
    )
    circular_flag = int(circular_count > 0)
    rapid_flag = int(state.rapid_count > 0)
    typology_flag_count = (
        fan_in_flag
        + fan_out_flag
        + structuring_flag
        + two_way_flag
        + three_way_flag
        + rapid_flag
    )

    values = np.asarray(
        [
            state.incoming_count,
            unique_senders,
            state.incoming_total,
            average_incoming,
            state.incoming_maximum,
            state.outgoing_count,
            unique_recipients,
            state.outgoing_total,
            average_outgoing,
            state.outgoing_maximum,
            fan_in_flag,
            fan_out_flag,
            state.near_threshold_count,
            state.maximum_near_threshold_daily_count,
            state.structuring_event_count,
            structuring_flag,
            state.two_account_cycle_count,
            state.three_account_cycle_count,
            two_way_flag,
            three_way_flag,
            circular_count,
            circular_flag,
            state.rapid_count,
            average_rapid_delay,
            minimum_rapid_delay,
            rapid_flag,
            typology_flag_count,
        ],
        dtype=np.float32,
    )

    for index, column in enumerate(BASE_TYPOLOGY_COLUMNS):
        if column in LOG_TRANSFORM_COLUMNS:
            values[index] = np.log1p(max(values[index], 0.0))
    return values


def build_point_in_time_features(
    transactions: pd.DataFrame,
    output_file: Path,
) -> np.memmap:
    """
    Build features chronologically.

    All rows with the same timestamp first receive features from the state that
    existed before that timestamp. Only afterwards are those transactions added
    to the state. Thus no transaction can see itself, another simultaneous row,
    or a future transaction.
    """
    row_count = len(transactions)
    feature_count = len(TYPOLOGY_FEATURE_COLUMNS)
    matrix = np.lib.format.open_memmap(
        output_file,
        mode="w+",
        dtype="float32",
        shape=(row_count, feature_count),
    )

    sender_nodes, recipient_nodes, account_count = encode_accounts(transactions)
    print(f"Accounts represented in temporal state: {account_count:,}")

    timestamps = transactions["timestamp"].to_numpy(dtype="datetime64[ns]")
    timestamps_ns = timestamps.astype("int64", copy=False)
    amount_paid = (
        pd.to_numeric(transactions["amount_paid"], errors="coerce")
        .fillna(0)
        .clip(lower=0)
        .to_numpy(dtype=np.float64)
    )
    amount_received = (
        pd.to_numeric(transactions["amount_received"], errors="coerce")
        .fillna(0)
        .clip(lower=0)
        .to_numpy(dtype=np.float64)
    )
    currencies, _ = pd.factorize(
        transactions["payment_currency"].astype("string").fillna("<missing>"),
        sort=False,
    )
    days = transactions["timestamp"].dt.floor("D").to_numpy(dtype="datetime64[D]")
    days_integer = days.astype("int64", copy=False)

    states: dict[int, AccountState] = {}
    outgoing_neighbors: dict[int, set[int]] = defaultdict(set)
    incoming_neighbors: dict[int, set[int]] = defaultdict(set)
    near_daily_counts: dict[tuple[int, int, int], int] = defaultdict(int)
    recent_incoming: dict[int, deque[tuple[int, float]]] = defaultdict(deque)
    rapid_window_ns = int(pd.Timedelta(hours=RAPID_HOURS).value)

    started = time.perf_counter()
    start = 0
    while start < row_count:
        end = start + 1
        timestamp_ns = timestamps_ns[start]
        while end < row_count and timestamps_ns[end] == timestamp_ns:
            end += 1

        # Phase 1: record features before any row at this timestamp is processed.
        for row_index in range(start, end):
            sender = int(sender_nodes[row_index])
            recipient = int(recipient_nodes[row_index])
            matrix[row_index, : len(BASE_TYPOLOGY_COLUMNS)] = state_values(
                states.get(sender)
            )
            matrix[row_index, len(BASE_TYPOLOGY_COLUMNS) :] = state_values(
                states.get(recipient)
            )

        # Phase 2: detect events against strictly earlier history, then update state.
        for row_index in range(start, end):
            sender = int(sender_nodes[row_index])
            recipient = int(recipient_nodes[row_index])
            paid = float(amount_paid[row_index])
            received = float(amount_received[row_index])

            sender_state = states.setdefault(sender, AccountState())
            recipient_state = states.setdefault(recipient, AccountState())

            # Rapid pass-through: compare the outgoing payment with the most
            # recent earlier incoming payment to the sender.
            queue = recent_incoming.get(sender)
            if queue:
                cutoff = timestamp_ns - rapid_window_ns
                while queue and queue[0][0] < cutoff:
                    queue.popleft()
                if queue:
                    incoming_time, incoming_amount = queue[-1]
                    if incoming_amount > 0:
                        ratio = paid / incoming_amount
                        if RAPID_MIN_RATIO <= ratio <= RAPID_MAX_RATIO:
                            delay_hours = (
                                timestamp_ns - incoming_time
                            ) / 3_600_000_000_000
                            sender_state.rapid_count += 1
                            sender_state.rapid_delay_total += delay_hours
                            sender_state.rapid_delay_minimum = min(
                                sender_state.rapid_delay_minimum, delay_hours
                            )

            # Structuring state for the sender.
            if STRUCTURING_LOWER_BOUND <= paid < STRUCTURING_THRESHOLD:
                sender_state.near_threshold_count += 1
                daily_key = (
                    sender,
                    int(currencies[row_index]),
                    int(days_integer[row_index]),
                )
                near_daily_counts[daily_key] += 1
                current_daily_count = near_daily_counts[daily_key]
                sender_state.maximum_near_threshold_daily_count = max(
                    sender_state.maximum_near_threshold_daily_count,
                    current_daily_count,
                )
                if current_daily_count == STRUCTURING_MIN_TRANSACTIONS:
                    sender_state.structuring_event_count += 1

            # A newly observed directed pair may complete reciprocal or
            # three-account circular paths. Repeated edges do not create a new
            # graph-structure cycle.
            if recipient not in outgoing_neighbors[sender]:
                if sender in outgoing_neighbors[recipient]:
                    sender_state.two_account_cycle_count += 1
                    recipient_state.two_account_cycle_count += 1

                triangle_nodes = (
                    outgoing_neighbors.get(recipient, set())
                    & incoming_neighbors.get(sender, set())
                )
                for third_node in triangle_nodes:
                    sender_state.three_account_cycle_count += 1
                    recipient_state.three_account_cycle_count += 1
                    states.setdefault(
                        int(third_node), AccountState()
                    ).three_account_cycle_count += 1

                outgoing_neighbors[sender].add(recipient)
                incoming_neighbors[recipient].add(sender)

            sender_state.outgoing_count += 1
            sender_state.outgoing_total += paid
            sender_state.outgoing_maximum = max(
                sender_state.outgoing_maximum, paid
            )
            sender_state.outgoing_recipients.add(recipient)

            recipient_state.incoming_count += 1
            recipient_state.incoming_total += received
            recipient_state.incoming_maximum = max(
                recipient_state.incoming_maximum, received
            )
            recipient_state.incoming_senders.add(sender)

        # Phase 3: only after processing the timestamp do its incoming payments
        # become available to later timestamps.
        for row_index in range(start, end):
            recipient = int(recipient_nodes[row_index])
            recent_incoming[recipient].append(
                (timestamp_ns, float(amount_received[row_index]))
            )

        if end % 250_000 < (end - start):
            elapsed = time.perf_counter() - started
            print(
                f"Point-in-time features: {end:,}/{row_count:,} rows "
                f"({100 * end / row_count:.1f}%, {elapsed:.1f}s)"
            )
            matrix.flush()
        start = end

    matrix.flush()
    del sender_nodes, recipient_nodes
    return matrix


def load_or_build_point_in_time_features(
    transactions: pd.DataFrame,
    rebuild: bool,
) -> np.memmap:
    expected_metadata = {
        "version": FEATURE_CACHE_VERSION,
        "rows": len(transactions),
        "columns": len(TYPOLOGY_FEATURE_COLUMNS),
        "first_timestamp": str(transactions["timestamp"].iloc[0]),
        "last_timestamp": str(transactions["timestamp"].iloc[-1]),
    }

    if (
        not rebuild
        and POINT_IN_TIME_FILE.exists()
        and POINT_IN_TIME_METADATA_FILE.exists()
    ):
        saved_metadata = json.loads(
            POINT_IN_TIME_METADATA_FILE.read_text(encoding="utf-8")
        )
        if saved_metadata == expected_metadata:
            print(f"Loading point-in-time features: {POINT_IN_TIME_FILE}")
            return np.load(POINT_IN_TIME_FILE, mmap_mode="r")

    print("Building leakage-free point-in-time typology features...")
    matrix = build_point_in_time_features(transactions, POINT_IN_TIME_FILE)
    POINT_IN_TIME_METADATA_FILE.write_text(
        json.dumps(expected_metadata, indent=2), encoding="utf-8"
    )
    return matrix


def attach_point_in_time_block(
    raw_features: pd.DataFrame,
    typology_block: np.ndarray,
) -> pd.DataFrame:
    typology_frame = pd.DataFrame(
        np.asarray(typology_block),
        index=raw_features.index,
        columns=TYPOLOGY_FEATURE_COLUMNS,
        dtype="float32",
    )
    return pd.concat([raw_features, typology_frame], axis=1)


def save_feature_importance(
    model_name: str,
    model: object,
    feature_names: list[str],
    output_dir: Path,
) -> None:
    importance = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv(
        output_dir / f"{model_name}_feature_importance.csv", index=False
    )


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


def model_from_parameters(
    model_name: str,
    parameters: dict[str, object],
    y_train: pd.Series,
) -> object:
    """Create one model while keeping the non-tuned settings fixed."""
    scale = y_train.eq(0).sum() / y_train.eq(1).sum()
    if model_name == "random_forest":
        return RandomForestClassifier(
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            **parameters,
        )
    if model_name == "xgboost":
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            scale_pos_weight=scale,
            **parameters,
        )
    raise ValueError(f"Unknown model: {model_name}")


def tune_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    tune_model: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    """
    Select hyperparameters using validation PR-AUC only.

    The test set is never used during tuning. The returned models are the
    fitted candidates with the best validation PR-AUC.
    """
    grids: dict[str, dict[str, list[object]]] = {
        "random_forest": {
            "n_estimators": [350],
            "max_depth": [12, 20],
            "min_samples_leaf": [1, 4],
            "max_features": ["sqrt", 0.5],
        },
        "xgboost": {
            "n_estimators": [500],
            "max_depth": [4, 8],
            "learning_rate": [0.03, 0.05],
            "min_child_weight": [1, 5],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
        },
    }
    selected_names = (
        ["random_forest", "xgboost"]
        if tune_model == "both"
        else [tune_model]
    )
    fitted_best: dict[str, object] = {}
    rows: list[dict[str, object]] = []

    for model_name in selected_names:
        best_score = -np.inf
        best_model: object | None = None
        best_parameters: dict[str, object] | None = None

        for candidate_number, parameters in enumerate(
            ParameterGrid(grids[model_name]), start=1
        ):
            print(
                f"Tuning {model_name}: candidate {candidate_number} "
                f"{parameters}"
            )
            model = model_from_parameters(model_name, parameters, y_train)
            started = time.perf_counter()
            model.fit(x_train, y_train)
            training_seconds = time.perf_counter() - started
            probabilities = model.predict_proba(x_validation)[:, 1]
            validation_pr_auc = average_precision_score(
                y_validation, probabilities
            )
            row = {
                "model": model_name,
                "candidate": candidate_number,
                "validation_pr_auc": float(validation_pr_auc),
                "training_seconds": training_seconds,
                **parameters,
            }
            rows.append(row)
            print(
                f"Validation PR-AUC: {validation_pr_auc:.6f} "
                f"({training_seconds:.1f}s)"
            )

            if validation_pr_auc > best_score:
                best_score = float(validation_pr_auc)
                best_model = model
                best_parameters = dict(parameters)

        if best_model is None or best_parameters is None:
            raise RuntimeError(f"No tuning candidate completed for {model_name}.")
        fitted_best[model_name] = best_model
        print(
            f"Best {model_name} validation PR-AUC: {best_score:.6f}; "
            f"parameters: {best_parameters}"
        )

    return fitted_best, pd.DataFrame(rows)


def save_tuning_comparison(tuned_results: pd.DataFrame) -> None:
    """Compare tuned results with the existing untuned typology benchmark."""
    frames: list[pd.DataFrame] = []
    untuned_file = OUTPUT_DIR / "benchmark_comparison.csv"
    if untuned_file.exists():
        untuned = pd.read_csv(untuned_file)
        untuned["configuration"] = "untuned"
        frames.append(untuned)

    tuned = tuned_results.copy()
    tuned["configuration"] = "tuned"
    frames.append(tuned)
    pd.concat(frames, ignore_index=True).to_csv(
        TUNING_OUTPUT_DIR / "untuned_vs_tuned_comparison.csv",
        index=False,
    )


def combine_comparisons(corrected_results: pd.DataFrame) -> None:
    frames: list[pd.DataFrame] = []

    raw_file = RAW_BENCHMARK_DIR / "benchmark_comparison.csv"
    if raw_file.exists():
        raw = pd.read_csv(raw_file)
        raw.insert(0, "feature_set", "transaction_only")
        frames.append(raw)

    corrected = corrected_results.copy()
    if "feature_set" not in corrected.columns:
        corrected.insert(0, "feature_set", "transaction_plus_point_in_time_typology")
    frames.append(corrected)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(COMBINED_COMPARISON_FILE, index=False)
    print(f"Combined comparison saved to: {COMBINED_COMPARISON_FILE}")


def main() -> None:
    args = parse_arguments()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading IBM transactions from: {args.input}")
    transactions = load_transactions(args.input)
    if not transactions["timestamp"].is_monotonic_increasing:
        print("Sorting transactions chronologically...")
        transactions = transactions.sort_values(
            "timestamp", kind="stable"
        ).reset_index(drop=True)
    transactions = transactions.rename(columns={"is_laundering": "target"})

    train_end = int(len(transactions) * TRAIN_FRACTION)
    validation_end = int(
        len(transactions) * (TRAIN_FRACTION + VALIDATION_FRACTION)
    )
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
        raise ValueError(
            "At least one chronological split has no laundering transactions."
        )

    typology_matrix = load_or_build_point_in_time_features(
        transactions, args.rebuild_features
    )

    frequency_maps = create_frequency_maps(train)
    print("Creating transaction and point-in-time AML typology features...")
    x_train = attach_point_in_time_block(
        create_features(train, frequency_maps),
        typology_matrix[:train_end],
    )
    x_validation = attach_point_in_time_block(
        create_features(validation, frequency_maps),
        typology_matrix[train_end:validation_end],
    )
    x_test = attach_point_in_time_block(
        create_features(test, frequency_maps),
        typology_matrix[validation_end:],
    )

    y_train = train["target"].astype("int8")
    y_validation = validation["target"].astype("int8")
    y_test = test["target"].astype("int8")
    del train, validation, test, transactions, typology_matrix

    x_train, y_train_sampled = sample_training_data(x_train, y_train)
    print(
        f"Sampled training rows: {len(x_train):,} "
        f"({int(y_train_sampled.sum()):,} laundering)"
    )

    run_output_dir = TUNING_OUTPUT_DIR if args.tune else OUTPUT_DIR
    run_output_dir.mkdir(parents=True, exist_ok=True)

    if args.tune:
        models, tuning_search = tune_models(
            x_train,
            y_train_sampled,
            x_validation,
            y_validation,
            args.tune_model,
        )
        tuning_search.to_csv(
            run_output_dir / "validation_hyperparameter_search.csv",
            index=False,
        )
    else:
        models = create_models(y_train_sampled)

    results: list[dict[str, object]] = []
    for model_name, model in models.items():
        print(f"\nTraining {model_name} with point-in-time typology features...")
        if args.tune:
            # The best tuning candidate is already fitted on the training set.
            training_seconds = float(
                tuning_search.loc[
                    (tuning_search["model"] == model_name)
                    & (
                        tuning_search["validation_pr_auc"]
                        == tuning_search.loc[
                            tuning_search["model"] == model_name,
                            "validation_pr_auc",
                        ].max()
                    ),
                    "training_seconds",
                ].iloc[0]
            )
        else:
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
        metrics["feature_set"] = (
            "transaction_plus_point_in_time_typology_tuned"
            if args.tune
            else "transaction_plus_point_in_time_typology"
        )
        results.append(metrics)
        print(json.dumps(metrics, indent=2))

        (run_output_dir / f"{model_name}_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        save_feature_importance(
            model_name, model, list(x_train.columns), run_output_dir
        )
        if args.save_models:
            joblib.dump(
                model, run_output_dir / f"{model_name}.joblib", compress=3
            )

    comparison = pd.DataFrame(results)
    comparison.to_csv(run_output_dir / "benchmark_comparison.csv", index=False)
    if args.tune:
        save_tuning_comparison(comparison)
        print(f"Tuned benchmark saved to: {run_output_dir}")
    else:
        combine_comparisons(comparison)
        print(f"Corrected benchmark saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
