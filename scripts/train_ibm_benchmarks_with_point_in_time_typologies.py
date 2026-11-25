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
from sklearn.model_selection import ParameterGrid, ParameterSampler

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

TYPOLOGY_GROUPS: dict[str, list[str]] = {
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
    "all": BASE_TYPOLOGY_COLUMNS,
}

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
    parser.add_argument(
        "--typology",
        choices=tuple(TYPOLOGY_GROUPS),
        default="all",
        help=(
            "Typology feature group to add to the transaction features. "
            "Use transaction_only for no typology features (default: all)."
        ),
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
    base_columns: list[str] | None = None,
) -> pd.DataFrame:
    if base_columns is None:
        base_columns = BASE_TYPOLOGY_COLUMNS
    if not base_columns:
        return raw_features.copy()

    selected_columns = [
        f"{role}_{column}"
        for role in ("sender", "recipient")
        for column in base_columns
    ]
    column_positions = {
        name: position
        for position, name in enumerate(TYPOLOGY_FEATURE_COLUMNS)
    }
    selected_positions = [
        column_positions[column] for column in selected_columns
    ]
    typology_frame = pd.DataFrame(
        np.asarray(typology_block[:, selected_positions]),
        index=raw_features.index,
        columns=selected_columns,
        dtype="float32",
    )
    return pd.concat([raw_features, typology_frame], axis=1)


def build_test_context(test: pd.DataFrame) -> pd.DataFrame:
    """Keep the identifiers needed for transaction-level explanations."""
    identifier_column = next(
        (
            column
            for column in ("transaction_id", "tran_id", "tx_id", "id")
            if column in test.columns
        ),
        None,
    )
    context = pd.DataFrame(index=test.index)
    if identifier_column is None:
        context["transaction_id"] = test.index.astype(str)
    else:
        context["transaction_id"] = test[identifier_column].astype(str)
    context["sender"] = test["sender_id"].astype(str)
    context["recipient"] = test["recipient_id"].astype(str)
    context["actual_is_laundering"] = test["target"].astype(bool)
    return context.reset_index(drop=True)


def original_typology_value(
    feature_row: pd.Series, role: str, base_column: str
) -> float:
    value = float(feature_row[f"{role}_{base_column}"])
    if base_column in LOG_TRANSFORM_COLUMNS:
        value = float(np.expm1(value))
    return max(value, 0.0)


def build_prediction_explanations(
    model_name: str,
    test_context: pd.DataFrame,
    test_probabilities: np.ndarray,
    threshold: float,
    typology_block: np.ndarray,
) -> pd.DataFrame:
    """Explain transactions predicted as laundering using existing rule flags."""
    predicted = test_probabilities >= threshold
    positions = np.flatnonzero(predicted)
    columns = [
        "transaction_id",
        "sender",
        "recipient",
        "probability",
        "predicted_is_laundering",
        "actual_is_laundering",
        "prediction_correct",
        "detected_typologies",
        "fan_in_evidence",
        "fan_out_evidence",
        "structuring_evidence",
        "circular_transfer_evidence",
        "rapid_pass_through_evidence",
    ]
    if len(positions) == 0:
        return pd.DataFrame(columns=columns)

    selected_context = test_context.iloc[positions].reset_index(drop=True)
    selected_features = pd.DataFrame(
        np.asarray(typology_block[positions]),
        columns=TYPOLOGY_FEATURE_COLUMNS,
        dtype="float32",
    )

    rows: list[dict[str, object]] = []
    for local_index, feature_row in selected_features.iterrows():
        typologies: list[str] = []
        evidence: dict[str, list[str]] = {
            "fan_in": [],
            "fan_out": [],
            "structuring": [],
            "circular": [],
            "rapid": [],
        }

        for role in ("sender", "recipient"):
            if original_typology_value(
                feature_row, role, "fan_in_flag"
            ) >= 0.5:
                typologies.append("Fan-in")
                senders = round(
                    original_typology_value(
                        feature_row, role, "unique_senders"
                    )
                )
                incoming = round(
                    original_typology_value(
                        feature_row, role, "incoming_transaction_count"
                    )
                )
                evidence["fan_in"].append(
                    f"{role}: {senders} unique senders, "
                    f"{incoming} earlier incoming transactions"
                )

            if original_typology_value(
                feature_row, role, "fan_out_flag"
            ) >= 0.5:
                typologies.append("Fan-out")
                recipients = round(
                    original_typology_value(
                        feature_row, role, "unique_recipients"
                    )
                )
                outgoing = round(
                    original_typology_value(
                        feature_row, role, "outgoing_transaction_count"
                    )
                )
                evidence["fan_out"].append(
                    f"{role}: {recipients} unique recipients, "
                    f"{outgoing} earlier outgoing transactions"
                )

            if original_typology_value(
                feature_row, role, "structuring_flag"
            ) >= 0.5:
                typologies.append("Structuring")
                near_count = round(
                    original_typology_value(
                        feature_row, role, "near_threshold_transaction_count"
                    )
                )
                daily_maximum = round(
                    original_typology_value(
                        feature_row,
                        role,
                        "maximum_near_threshold_daily_count",
                    )
                )
                events = round(
                    original_typology_value(
                        feature_row, role, "structuring_event_count"
                    )
                )
                evidence["structuring"].append(
                    f"{role}: {near_count} earlier near-threshold transactions, "
                    f"daily maximum {daily_maximum}, {events} events"
                )

            if original_typology_value(
                feature_row, role, "circular_transfer_flag"
            ) >= 0.5:
                typologies.append("Circular transfer")
                two_way = round(
                    original_typology_value(
                        feature_row, role, "two_account_cycle_count"
                    )
                )
                three_way = round(
                    original_typology_value(
                        feature_row, role, "three_account_cycle_count"
                    )
                )
                evidence["circular"].append(
                    f"{role}: {two_way} two-account and "
                    f"{three_way} three-account earlier cycles"
                )

            if original_typology_value(
                feature_row, role, "rapid_pass_through_flag"
            ) >= 0.5:
                typologies.append("Rapid pass-through")
                count = round(
                    original_typology_value(
                        feature_row, role, "rapid_pass_through_count"
                    )
                )
                average_delay = original_typology_value(
                    feature_row, role, "average_rapid_pass_through_hours"
                )
                minimum_delay = original_typology_value(
                    feature_row, role, "minimum_rapid_pass_through_hours"
                )
                evidence["rapid"].append(
                    f"{role}: {count} earlier events, average delay "
                    f"{average_delay:.2f} h, minimum delay {minimum_delay:.2f} h"
                )

        context_row = selected_context.iloc[local_index]
        actual = bool(context_row["actual_is_laundering"])
        rows.append(
            {
                "transaction_id": context_row["transaction_id"],
                "sender": context_row["sender"],
                "recipient": context_row["recipient"],
                "probability": float(test_probabilities[positions[local_index]]),
                "predicted_is_laundering": True,
                "actual_is_laundering": actual,
                "prediction_correct": actual,
                "detected_typologies": "; ".join(
                    dict.fromkeys(typologies)
                ) or "No rule-based typology flag",
                "fan_in_evidence": "; ".join(evidence["fan_in"]),
                "fan_out_evidence": "; ".join(evidence["fan_out"]),
                "structuring_evidence": "; ".join(
                    evidence["structuring"]
                ),
                "circular_transfer_evidence": "; ".join(
                    evidence["circular"]
                ),
                "rapid_pass_through_evidence": "; ".join(
                    evidence["rapid"]
                ),
            }
        )
    result = pd.DataFrame(rows, columns=columns)
    return result.sort_values("probability", ascending=False).reset_index(
        drop=True
    )


def build_prediction_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize predicted laundering transactions and detected typologies."""
    total = len(predictions)
    correct = int(predictions["prediction_correct"].sum()) if total else 0
    incorrect = total - correct

    def evidence_count(column: str) -> int:
        if total == 0:
            return 0
        return int(predictions[column].fillna("").astype(str).str.strip().ne("").sum())

    no_typology = 0
    if total:
        no_typology = int(
            predictions["detected_typologies"]
            .eq("No rule-based typology flag")
            .sum()
        )

    rows = [
        {
            "metric": "Predicted laundering transactions",
            "count": total,
            "description": "Transactions whose probability met or exceeded the selected threshold.",
        },
        {
            "metric": "Correct laundering predictions",
            "count": correct,
            "description": "Predicted laundering and actually labeled as laundering (true positives).",
        },
        {
            "metric": "Incorrect laundering predictions",
            "count": incorrect,
            "description": "Predicted laundering but actually labeled as normal (false positives).",
        },
        {
            "metric": "Fan-in detected",
            "count": evidence_count("fan_in_evidence"),
            "description": "Predicted transactions with point-in-time fan-in evidence.",
        },
        {
            "metric": "Fan-out detected",
            "count": evidence_count("fan_out_evidence"),
            "description": "Predicted transactions with point-in-time fan-out evidence.",
        },
        {
            "metric": "Structuring detected",
            "count": evidence_count("structuring_evidence"),
            "description": "Predicted transactions with point-in-time structuring evidence.",
        },
        {
            "metric": "Circular transfer detected",
            "count": evidence_count("circular_transfer_evidence"),
            "description": "Predicted transactions with point-in-time circular-transfer evidence.",
        },
        {
            "metric": "Rapid pass-through detected",
            "count": evidence_count("rapid_pass_through_evidence"),
            "description": "Predicted transactions with point-in-time rapid pass-through evidence.",
        },
        {
            "metric": "No typology flag detected",
            "count": no_typology,
            "description": "Predicted transactions without any of the five rule-based typology flags.",
        },
    ]
    summary = pd.DataFrame(rows)
    summary["percentage_of_predicted"] = (
        summary["count"] / total if total else 0.0
    )
    return summary[
        ["metric", "count", "percentage_of_predicted", "description"]
    ]


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
    The test set is not used during tuning.
    """

    grids: dict[str, dict[str, list[object]]] = {
        "random_forest": {
            "n_estimators": [500, 800],
            "max_depth": [16, 24, None],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", 0.3, 0.5],
            "criterion": ["gini", "entropy"],
            "class_weight": [
                "balanced",
                "balanced_subsample",
                {0: 1, 1: 5},
                {0: 1, 1: 10},
            ],
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

        if model_name == "random_forest":
            candidates = list(
                ParameterSampler(
                    grids[model_name],
                    n_iter=40,
                    random_state=RANDOM_STATE,
                )
            )
        else:
            candidates = list(ParameterGrid(grids[model_name]))

        for candidate_number, parameters in enumerate(
            candidates,
            start=1,
        ):
            print(
                f"Tuning {model_name}: candidate {candidate_number}/{len(candidates)} "
                f"{parameters}"
            )

            model = model_from_parameters(
                model_name,
                parameters,
                y_train,
            )

            started = time.perf_counter()
            model.fit(x_train, y_train)
            training_seconds = time.perf_counter() - started

            validation_probabilities = model.predict_proba(
                x_validation
            )[:, 1]

            validation_pr_auc = average_precision_score(
                y_validation,
                validation_probabilities,
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
            raise RuntimeError(
                f"No tuning candidate completed for {model_name}."
            )

        fitted_best[model_name] = best_model

        print(
            f"Best {model_name} validation PR-AUC: "
            f"{best_score:.6f}; parameters: {best_parameters}"
        )

    return fitted_best, pd.DataFrame(rows)


def save_tuning_comparison(
    tuned_results: pd.DataFrame, output_dir: Path
) -> None:
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
        output_dir / "untuned_vs_tuned_comparison.csv",
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


def untuned_typology_result_file(typology: str) -> Path:
    if typology == "all":
        return OUTPUT_DIR / "benchmark_comparison.csv"
    return (
        OUTPUT_DIR
        / "single_typology_runs"
        / typology
        / "benchmark_comparison.csv"
    )


def load_transaction_only_results() -> pd.DataFrame:
    """Find the existing transaction-only results and return both models."""
    candidate_files = [
        RAW_BENCHMARK_DIR / "benchmark_comparison.csv",
        IBM_OUTPUT_DIR
        / "benchmark_4_typology_ablation"
        / "ablation_comparison.csv",
        COMBINED_COMPARISON_FILE,
        IBM_OUTPUT_DIR / "benchmark_1_vs_2_comparison.csv",
    ]
    for candidate in candidate_files:
        if not candidate.exists():
            continue
        results = pd.read_csv(candidate)
        if "feature_set" in results.columns:
            results = results.loc[
                results["feature_set"].astype(str) == "transaction_only"
            ].copy()
        if "model" not in results.columns:
            continue
        results = results.loc[
            results["model"].isin(["random_forest", "xgboost"])
        ].drop_duplicates(subset="model", keep="first")
        if set(results["model"]) == {"random_forest", "xgboost"}:
            results["feature_set"] = "transaction_only"
            results["configuration"] = "untuned"
            return results

    checked = "\n".join(f"  - {path}" for path in candidate_files)
    raise FileNotFoundError(
        "The Excel comparison requires transaction-only results for both "
        "Random Forest and XGBoost. None were found in:\n"
        f"{checked}\nRun train_ibm_benchmarks.py or the ablation script first."
    )


def best_parameter_text(
    model_name: str, tuning_search: pd.DataFrame | None
) -> str:
    if tuning_search is None:
        return ""
    candidates = tuning_search.loc[
        tuning_search["model"] == model_name
    ].copy()
    if candidates.empty:
        return ""
    best = candidates.loc[candidates["validation_pr_auc"].idxmax()]
    excluded = {
        "model",
        "candidate",
        "validation_pr_auc",
        "training_seconds",
    }
    parameters = [
        f"{name}={value}"
        for name, value in best.items()
        if name not in excluded and pd.notna(value)
    ]
    return "; ".join(parameters)


def save_excel_comparison_report(
    current_results: pd.DataFrame,
    typology: str,
    tuned: bool,
    output_dir: Path,
    tuning_search: pd.DataFrame | None = None,
    prediction_explanations: dict[str, pd.DataFrame] | None = None,
) -> None:
    """Save model/feature comparisons and RF tuning details in one workbook."""
    metric_columns = [
        "threshold",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "false_positive_rate",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
        "training_seconds",
        "inference_seconds",
    ]
    transaction_only = load_transaction_only_results()

    current = current_results.copy()
    current["feature_set"] = (
        "transaction_only"
        if typology == "transaction_only"
        else f"transaction_plus_{typology}"
    )
    current["configuration"] = "tuned" if tuned else "untuned"

    required_models = {"random_forest", "xgboost"}
    if set(current["model"]) != required_models:
        raise ValueError(
            "The four-row Excel comparison requires both models. Run tuning "
            "with --tune-model both."
        )

    model_order = {"random_forest": 0, "xgboost": 1}
    transaction_only["_model_order"] = transaction_only["model"].map(
        model_order
    )
    current["_model_order"] = current["model"].map(model_order)
    transaction_only = transaction_only.sort_values("_model_order")
    current = current.sort_values("_model_order")
    model_comparison = pd.concat(
        [transaction_only, current], ignore_index=True
    ).drop(columns="_model_order")

    comparison_metrics = [
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "false_positive_rate",
        "false_positives",
        "false_negatives",
        "true_positives",
    ]
    baseline_by_model = transaction_only.set_index("model")
    improved_column: list[str] = []
    declined_column: list[str] = []
    selection_metric_column: list[str] = []
    threshold_metric_column: list[str] = []
    parameter_column: list[str] = []

    for _, row in model_comparison.iterrows():
        is_baseline = row["feature_set"] == "transaction_only"
        model_name = str(row["model"])
        if is_baseline:
            improved_column.append("baseline")
            declined_column.append("baseline")
            selection_metric_column.append("not tuned in this run")
            threshold_metric_column.append("validation F1")
            parameter_column.append("")
            continue

        baseline = baseline_by_model.loc[model_name]
        improved: list[str] = []
        declined: list[str] = []
        for metric in comparison_metrics:
            if metric not in row.index or metric not in baseline.index:
                continue
            difference = float(row[metric]) - float(baseline[metric])
            lower_is_better = metric in {
                "false_positive_rate",
                "false_positives",
                "false_negatives",
            }
            if np.isclose(difference, 0.0):
                continue
            if (difference < 0) == lower_is_better:
                improved.append(metric)
            else:
                declined.append(metric)
        improved_column.append(", ".join(improved) or "none")
        declined_column.append(", ".join(declined) or "none")
        selection_metric_column.append(
            "validation PR-AUC" if tuned else "not tuned in this run"
        )
        threshold_metric_column.append("validation F1")
        parameter_column.append(
            best_parameter_text(model_name, tuning_search) if tuned else ""
        )

    model_comparison["hyperparameter_selection_metric"] = (
        selection_metric_column
    )
    model_comparison["threshold_selection_metric"] = threshold_metric_column
    model_comparison["selected_hyperparameters"] = parameter_column
    model_comparison["improved_test_metrics_vs_transaction_only"] = (
        improved_column
    )
    model_comparison["declined_test_metrics_vs_transaction_only"] = (
        declined_column
    )
    preferred_columns = [
        column
        for column in [
            "feature_set",
            "configuration",
            "model",
            "hyperparameter_selection_metric",
            "threshold_selection_metric",
            "selected_hyperparameters",
            *metric_columns,
            "improved_test_metrics_vs_transaction_only",
            "declined_test_metrics_vs_transaction_only",
        ]
        if column in model_comparison.columns
    ]
    remaining_columns = [
        column
        for column in model_comparison.columns
        if column not in preferred_columns
    ]
    model_comparison = model_comparison[
        preferred_columns + remaining_columns
    ]

    rf_summary_rows: list[dict[str, object]] = [
        {
            "section": "method",
            "item": "Hyperparameter selection criterion",
            "value": "Highest validation PR-AUC",
            "comparison_value": "",
            "difference": "",
            "assessment": "",
        },
        {
            "section": "method",
            "item": "Classification-threshold criterion",
            "value": "Highest validation F1",
            "comparison_value": "",
            "difference": "",
            "assessment": "",
        },
    ]

    if tuned and tuning_search is not None:
        rf_search = tuning_search.loc[
            tuning_search["model"] == "random_forest"
        ].copy()
        if not rf_search.empty:
            best_rf = rf_search.loc[rf_search["validation_pr_auc"].idxmax()]
            excluded = {
                "model",
                "candidate",
                "validation_pr_auc",
                "training_seconds",
            }
            for parameter, value in best_rf.items():
                if parameter not in excluded and pd.notna(value):
                    rf_summary_rows.append(
                        {
                            "section": "best_hyperparameter",
                            "item": parameter,
                            "value": value,
                            "comparison_value": "",
                            "difference": "",
                            "assessment": "selected",
                        }
                    )
            rf_summary_rows.append(
                {
                    "section": "validation",
                    "item": "Best validation PR-AUC",
                    "value": float(best_rf["validation_pr_auc"]),
                    "comparison_value": "",
                    "difference": "",
                    "assessment": "selection score",
                }
            )

            untuned_file = untuned_typology_result_file(typology)
            if untuned_file.exists():
                untuned_results = pd.read_csv(untuned_file)
                untuned_rf = untuned_results.loc[
                    untuned_results["model"] == "random_forest"
                ]
                tuned_rf = current_results.loc[
                    current_results["model"] == "random_forest"
                ]
                if not untuned_rf.empty and not tuned_rf.empty:
                    untuned_rf = untuned_rf.iloc[0]
                    tuned_rf = tuned_rf.iloc[0]
                    for metric in metric_columns:
                        if (
                            metric not in untuned_rf.index
                            or metric not in tuned_rf.index
                        ):
                            continue
                        old_value = float(untuned_rf[metric])
                        new_value = float(tuned_rf[metric])
                        difference = new_value - old_value
                        lower_is_better = metric in {
                            "false_positive_rate",
                            "false_positives",
                            "false_negatives",
                            "training_seconds",
                            "inference_seconds",
                        }
                        improved = (
                            difference < 0 if lower_is_better else difference > 0
                        )
                        unchanged = np.isclose(difference, 0.0)
                        assessment = (
                            "unchanged"
                            if unchanged
                            else "improved" if improved else "declined"
                        )
                        rf_summary_rows.append(
                            {
                                "section": "test_metric_comparison",
                                "item": metric,
                                "value": old_value,
                                "comparison_value": new_value,
                                "difference": difference,
                                "assessment": assessment,
                            }
                        )
            else:
                rf_summary_rows.append(
                    {
                        "section": "test_metric_comparison",
                        "item": "Untuned comparison unavailable",
                        "value": str(untuned_file),
                        "comparison_value": "Run the same typology without --tune first.",
                        "difference": "",
                        "assessment": "not calculated",
                    }
                )
        else:
            rf_summary_rows.append(
                {
                    "section": "tuning",
                    "item": "Random Forest",
                    "value": "Not included in this tuning run",
                    "comparison_value": "Use --tune-model both or random_forest",
                    "difference": "",
                    "assessment": "not calculated",
                }
            )
    else:
        rf_summary_rows.append(
            {
                "section": "tuning",
                "item": "Random Forest",
                "value": "This was not a tuning run",
                "comparison_value": "Run again with --tune",
                "difference": "",
                "assessment": "not calculated",
            }
        )

    rf_summary = pd.DataFrame(rf_summary_rows)
    excel_file = output_dir / "transaction_vs_typology_model_comparison.xlsx"
    try:
        with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
            sheet_name = "comparison_report"
            model_start_row = 1
            model_comparison.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
                startrow=model_start_row,
            )

            rf_title_row = model_start_row + len(model_comparison) + 3
            rf_summary.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
                startrow=rf_title_row + 1,
            )

            validation_title_row: int | None = None
            if tuning_search is not None:
                validation_title_row = (
                    rf_title_row + len(rf_summary) + 4
                )
                tuning_search.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    index=False,
                    startrow=validation_title_row + 1,
                )

            worksheet = writer.book[sheet_name]
            worksheet.cell(1, 1, "MODEL AND FEATURE-SET COMPARISON")
            worksheet.cell(
                rf_title_row + 1,
                1,
                "RANDOM FOREST TUNING AND METRIC SUMMARY",
            )
            if validation_title_row is not None:
                worksheet.cell(
                    validation_title_row + 1,
                    1,
                    "VALIDATION HYPERPARAMETER SEARCH",
                )

            worksheet.freeze_panes = "A3"
            model_header_row = model_start_row + 1
            model_last_row = model_header_row + len(model_comparison)
            model_last_column = len(model_comparison.columns)
            worksheet.auto_filter.ref = (
                f"A{model_header_row}:"
                f"{worksheet.cell(model_header_row, model_last_column).column_letter}"
                f"{model_last_row}"
            )

            for column_cells in worksheet.columns:
                width = min(
                    max(
                        len(str(cell.value)) if cell.value is not None else 0
                        for cell in column_cells
                    )
                    + 2,
                    45,
                )
                worksheet.column_dimensions[
                    column_cells[0].column_letter
                ].width = width

            # Keep the existing comparison report and add one detailed
            # transaction-level worksheet for each model.
            if prediction_explanations:
                prediction_sheet_names = {
                    "random_forest": "random_forest_predictions",
                    "xgboost": "xgboost_predictions",
                }
                for model_name, predictions in prediction_explanations.items():
                    prediction_sheet = prediction_sheet_names.get(
                        model_name, f"{model_name[:22]}_predictions"
                    )
                    predictions.to_excel(
                        writer,
                        sheet_name=prediction_sheet,
                        index=False,
                    )
                    prediction_summary = build_prediction_summary(predictions)
                    summary_title_row = len(predictions) + 3
                    prediction_summary.to_excel(
                        writer,
                        sheet_name=prediction_sheet,
                        index=False,
                        startrow=summary_title_row + 1,
                    )
                    detail_worksheet = writer.book[prediction_sheet]
                    detail_worksheet.cell(
                        summary_title_row + 1,
                        1,
                        "PREDICTION AND TYPOLOGY SUMMARY",
                    )
                    summary_percentage_column = (
                        prediction_summary.columns.get_loc(
                            "percentage_of_predicted"
                        )
                        + 1
                    )
                    summary_header_row = summary_title_row + 2
                    for percentage_cell in detail_worksheet.iter_cols(
                        min_col=summary_percentage_column,
                        max_col=summary_percentage_column,
                        min_row=summary_header_row + 1,
                        max_row=summary_header_row + len(prediction_summary),
                    ):
                        for cell in percentage_cell:
                            cell.number_format = "0.00%"
                    detail_worksheet.freeze_panes = "A2"
                    if len(predictions) > 0:
                        last_column = detail_worksheet.cell(
                            1, len(predictions.columns)
                        ).column_letter
                        detail_worksheet.auto_filter.ref = (
                            f"A1:{last_column}{len(predictions) + 1}"
                        )
                    for column_cells in detail_worksheet.columns:
                        width = min(
                            max(
                                len(str(cell.value))
                                if cell.value is not None
                                else 0
                                for cell in column_cells
                            )
                            + 2,
                            55,
                        )
                        detail_worksheet.column_dimensions[
                            column_cells[0].column_letter
                        ].width = width

                    if len(predictions) > 0:
                        probability_column = predictions.columns.get_loc(
                            "probability"
                        ) + 1
                        for cell in detail_worksheet.iter_cols(
                            min_col=probability_column,
                            max_col=probability_column,
                            min_row=2,
                            max_row=len(predictions) + 1,
                        ):
                            for probability_cell in cell:
                                probability_cell.number_format = "0.000000"
    except ImportError as error:
        raise ImportError(
            "Excel output requires openpyxl. Install it with: "
            "python -m pip install --user openpyxl"
        ) from error
    print(f"Excel comparison saved to: {excel_file}")


def main() -> None:
    args = parse_arguments()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Create this directory even before tuning starts, making its location
    # visible and predictable. Actual results are written only with --tune.
    TUNING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    selected_typology_columns = TYPOLOGY_GROUPS[args.typology]
    print(
        "Creating transaction and point-in-time AML features "
        f"for selection: {args.typology}"
    )
    x_train = attach_point_in_time_block(
        create_features(train, frequency_maps),
        typology_matrix[:train_end],
        selected_typology_columns,
    )
    x_validation = attach_point_in_time_block(
        create_features(validation, frequency_maps),
        typology_matrix[train_end:validation_end],
        selected_typology_columns,
    )
    x_test = attach_point_in_time_block(
        create_features(test, frequency_maps),
        typology_matrix[validation_end:],
        selected_typology_columns,
    )

    # Retain the original test identifiers and the complete point-in-time
    # typology block for the transaction-level prediction worksheets.
    test_context = build_test_context(test)
    test_typology_block = typology_matrix[validation_end:]

    y_train = train["target"].astype("int8")
    y_validation = validation["target"].astype("int8")
    y_test = test["target"].astype("int8")
    del train, validation, test, transactions

    x_train, y_train_sampled = sample_training_data(x_train, y_train)
    print(
        f"Sampled training rows: {len(x_train):,} "
        f"({int(y_train_sampled.sum()):,} laundering)"
    )

    if args.tune:
        run_output_dir = TUNING_OUTPUT_DIR / args.typology
    elif args.typology == "all":
        run_output_dir = OUTPUT_DIR
    else:
        run_output_dir = OUTPUT_DIR / "single_typology_runs" / args.typology
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
    prediction_explanations: dict[str, pd.DataFrame] = {}
    for model_name, model in models.items():
        print(
            f"\nTraining {model_name} with feature selection: "
            f"{args.typology}..."
        )
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

        prediction_explanations[model_name] = build_prediction_explanations(
            model_name=model_name,
            test_context=test_context,
            test_probabilities=test_probabilities,
            threshold=threshold,
            typology_block=test_typology_block,
        )

        metrics = evaluate(
            model_name,
            y_test,
            test_probabilities,
            threshold,
            training_seconds,
            inference_seconds,
        )
        metrics["feature_set"] = args.typology
        metrics["tuned"] = bool(args.tune)
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
    save_excel_comparison_report(
        current_results=comparison,
        typology=args.typology,
        tuned=args.tune,
        output_dir=run_output_dir,
        tuning_search=tuning_search if args.tune else None,
        prediction_explanations=prediction_explanations,
    )
    del typology_matrix, test_typology_block
    if args.tune:
        save_tuning_comparison(comparison, run_output_dir)
        print(f"Tuned benchmark saved to: {run_output_dir}")
    else:
        combine_comparisons(comparison)
        print(f"Corrected benchmark saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
