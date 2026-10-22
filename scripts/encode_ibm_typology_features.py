from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
DEFAULT_INPUT = PROJECT_DIR / "IBM_PUBLIC_DATASET" / "HI-Small_Trans.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "ibm_aml"
FEATURE_FILE = OUTPUT_DIR / "ibm_account_features.csv"
RAPID_EVENT_FILE = OUTPUT_DIR / "ibm_rapid_pass_through_events.csv"

STRUCTURING_LOWER_BOUND = 1_000.0
STRUCTURING_THRESHOLD = 10_000.0
STRUCTURING_MIN_TRANSACTIONS = 3
FAN_IN_MIN_SENDERS = 5
FAN_OUT_MIN_RECIPIENTS = 5
RAPID_HOURS = 24
RAPID_MIN_RATIO = 0.80
RAPID_MAX_RATIO = 1.20


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode account-level AML typology features for IBM AML data."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    return parser.parse_args()


def load_transactions(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"IBM transaction file not found: {file_path}")

    data = pd.read_csv(
        file_path,
        dtype={
            "From Bank": "string",
            "Account": "string",
            "To Bank": "string",
            "Account.1": "string",
            "Receiving Currency": "category",
            "Payment Currency": "category",
            "Payment Format": "category",
            "Is Laundering": "int8",
        },
    )
    data = data.rename(
        columns={
            "Timestamp": "timestamp",
            "From Bank": "from_bank",
            "Account": "from_account",
            "To Bank": "to_bank",
            "Account.1": "to_account",
            "Amount Received": "amount_received",
            "Receiving Currency": "receiving_currency",
            "Amount Paid": "amount_paid",
            "Payment Currency": "payment_currency",
            "Payment Format": "payment_format",
            "Is Laundering": "is_laundering",
        }
    )

    required = {
        "timestamp", "from_bank", "from_account", "to_bank", "to_account",
        "amount_received", "receiving_currency", "amount_paid",
        "payment_currency", "payment_format", "is_laundering",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    data["transaction_id"] = np.arange(1, len(data) + 1, dtype=np.int64)
    data["sender_id"] = data["from_bank"].astype(str) + "::" + data["from_account"].astype(str)
    data["recipient_id"] = data["to_bank"].astype(str) + "::" + data["to_account"].astype(str)

    invalid = set(data["is_laundering"].dropna().unique()) - {0, 1}
    if invalid:
        raise ValueError(f"Invalid laundering labels: {invalid}")
    return data


def build_account_index(transactions: pd.DataFrame) -> pd.DataFrame:
    senders = transactions[["sender_id", "from_bank", "from_account"]].rename(
        columns={"sender_id": "account_key", "from_bank": "bank_id", "from_account": "account_id"}
    )
    recipients = transactions[["recipient_id", "to_bank", "to_account"]].rename(
        columns={"recipient_id": "account_key", "to_bank": "bank_id", "to_account": "account_id"}
    )
    accounts = pd.concat([senders, recipients], ignore_index=True)
    accounts = accounts.drop_duplicates("account_key").sort_values("account_key").reset_index(drop=True)
    accounts.insert(0, "node_index", np.arange(len(accounts), dtype=np.int64))
    return accounts


def add_basic_and_fan_features(
    features: pd.DataFrame, transactions: pd.DataFrame
) -> pd.DataFrame:
    incoming = transactions.groupby("recipient_id", observed=True).agg(
        incoming_transaction_count=("transaction_id", "size"),
        unique_senders=("sender_id", "nunique"),
        total_incoming_amount=("amount_received", "sum"),
        average_incoming_amount=("amount_received", "mean"),
        maximum_incoming_amount=("amount_received", "max"),
    )
    outgoing = transactions.groupby("sender_id", observed=True).agg(
        outgoing_transaction_count=("transaction_id", "size"),
        unique_recipients=("recipient_id", "nunique"),
        total_outgoing_amount=("amount_paid", "sum"),
        average_outgoing_amount=("amount_paid", "mean"),
        maximum_outgoing_amount=("amount_paid", "max"),
    )
    features = features.join(incoming, on="account_key").join(outgoing, on="account_key")
    features["fan_in_flag"] = (features["unique_senders"].fillna(0) >= FAN_IN_MIN_SENDERS).astype("int8")
    features["fan_out_flag"] = (features["unique_recipients"].fillna(0) >= FAN_OUT_MIN_RECIPIENTS).astype("int8")
    return features


def add_structuring_features(
    features: pd.DataFrame, transactions: pd.DataFrame
) -> pd.DataFrame:
    near = transactions.loc[
        transactions["amount_paid"].between(
            STRUCTURING_LOWER_BOUND, STRUCTURING_THRESHOLD, inclusive="left"
        ),
        ["sender_id", "timestamp", "payment_currency", "amount_paid"],
    ].copy()
    near["transaction_day"] = near["timestamp"].dt.floor("D")
    daily = near.groupby(
        ["sender_id", "payment_currency", "transaction_day"], observed=True
    ).agg(
        near_threshold_daily_count=("amount_paid", "size"),
        near_threshold_daily_amount=("amount_paid", "sum"),
    )
    daily["qualifies"] = (
        daily["near_threshold_daily_count"] >= STRUCTURING_MIN_TRANSACTIONS
    )
    structuring = daily.groupby("sender_id", observed=True).agg(
        near_threshold_transaction_count=("near_threshold_daily_count", "sum"),
        maximum_near_threshold_daily_count=("near_threshold_daily_count", "max"),
        structuring_event_count=("qualifies", "sum"),
    )
    features = features.join(structuring, on="account_key")
    features["structuring_flag"] = (features["structuring_event_count"].fillna(0) > 0).astype("int8")
    return features


def add_cycle_features(
    features: pd.DataFrame, transactions: pd.DataFrame, accounts: pd.DataFrame
) -> pd.DataFrame:
    index_map = pd.Series(accounts["node_index"].to_numpy(), index=accounts["account_key"])
    pairs = transactions[["sender_id", "recipient_id"]].drop_duplicates()
    rows = pairs["sender_id"].map(index_map).to_numpy(dtype=np.int64)
    cols = pairs["recipient_id"].map(index_map).to_numpy(dtype=np.int64)
    keep = rows != cols
    adjacency = sparse.csr_matrix(
        (np.ones(keep.sum(), dtype=np.int32), (rows[keep], cols[keep])),
        shape=(len(accounts), len(accounts)),
    )
    adjacency.data[:] = 1
    adjacency.eliminate_zeros()

    two_way = np.asarray(adjacency.multiply(adjacency.T).sum(axis=1)).ravel()
    two_way = two_way.astype(np.int64)

    print("Calculating directed three-account cycles; this may take several minutes...")
    two_step = adjacency @ adjacency
    three_way_matrix = two_step.multiply(adjacency.T)
    three_way = np.asarray(three_way_matrix.sum(axis=1)).ravel().astype(np.int64)
    del two_step, three_way_matrix, adjacency

    cycle_table = pd.DataFrame(
        {
            "node_index": accounts["node_index"],
            "two_account_cycle_count": two_way,
            "three_account_cycle_count": three_way,
        }
    )
    features = features.merge(cycle_table, on="node_index", how="left")
    features["two_way_circular_flag"] = (features["two_account_cycle_count"] > 0).astype("int8")
    features["three_way_circular_flag"] = (features["three_account_cycle_count"] > 0).astype("int8")
    features["circular_transfer_count"] = (
        features["two_account_cycle_count"] + features["three_account_cycle_count"]
    )
    features["circular_transfer_flag"] = (features["circular_transfer_count"] > 0).astype("int8")
    return features


def add_rapid_features(
    features: pd.DataFrame, transactions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    incoming = transactions[
        ["recipient_id", "transaction_id", "timestamp", "amount_received"]
    ].rename(
        columns={
            "recipient_id": "account_key",
            "transaction_id": "incoming_transaction_id",
            "timestamp": "incoming_timestamp",
            "amount_received": "incoming_amount",
        }
    ).sort_values("incoming_timestamp")
    outgoing = transactions[
        ["sender_id", "transaction_id", "timestamp", "amount_paid"]
    ].rename(
        columns={
            "sender_id": "account_key",
            "transaction_id": "outgoing_transaction_id",
            "timestamp": "outgoing_timestamp",
            "amount_paid": "outgoing_amount",
        }
    ).sort_values("outgoing_timestamp")

    matched = pd.merge_asof(
        outgoing,
        incoming,
        left_on="outgoing_timestamp",
        right_on="incoming_timestamp",
        by="account_key",
        direction="backward",
        tolerance=pd.Timedelta(hours=RAPID_HOURS),
        allow_exact_matches=False,
    )
    matched = matched.dropna(subset=["incoming_transaction_id", "incoming_amount"]).copy()
    matched["amount_ratio"] = matched["outgoing_amount"] / matched["incoming_amount"].replace(0, np.nan)
    matched = matched[matched["amount_ratio"].between(RAPID_MIN_RATIO, RAPID_MAX_RATIO)].copy()
    matched["delay_hours"] = (
        matched["outgoing_timestamp"] - matched["incoming_timestamp"]
    ).dt.total_seconds() / 3600

    rapid = matched.groupby("account_key", observed=True).agg(
        rapid_pass_through_count=("outgoing_transaction_id", "size"),
        average_rapid_pass_through_hours=("delay_hours", "mean"),
        minimum_rapid_pass_through_hours=("delay_hours", "min"),
    )
    features = features.join(rapid, on="account_key")
    features["rapid_pass_through_flag"] = (
        features["rapid_pass_through_count"].fillna(0) > 0
    ).astype("int8")
    return features, matched


def add_ground_truth_account_label(
    features: pd.DataFrame, transactions: pd.DataFrame
) -> pd.DataFrame:
    laundering = transactions[transactions["is_laundering"] == 1]
    labelled_accounts = pd.Index(
        pd.concat([laundering["sender_id"], laundering["recipient_id"]]).unique()
    )
    features["is_laundering_account"] = features["account_key"].isin(labelled_accounts).astype("int8")
    return features


def main() -> None:
    args = parse_arguments()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading IBM transactions from: {args.input}")
    transactions = load_transactions(args.input)
    accounts = build_account_index(transactions)
    features = accounts.copy()

    print("Calculating basic, fan-in and fan-out features...")
    features = add_basic_and_fan_features(features, transactions)
    print("Calculating structuring features...")
    features = add_structuring_features(features, transactions)
    print("Calculating two-way and three-way circular-transfer features...")
    features = add_cycle_features(features, transactions, accounts)
    print("Calculating rapid pass-through features...")
    features, rapid_events = add_rapid_features(features, transactions)
    features = add_ground_truth_account_label(features, transactions)

    numeric_columns = features.select_dtypes(include=["number"]).columns
    features[numeric_columns] = features[numeric_columns].fillna(0)
    flag_columns = [
        "structuring_flag", "fan_in_flag", "fan_out_flag",
        "two_way_circular_flag", "three_way_circular_flag",
        "rapid_pass_through_flag",
    ]
    features["typology_flag_count"] = features[flag_columns].sum(axis=1).astype("int8")

    features.to_csv(FEATURE_FILE, index=False)
    rapid_events.to_csv(RAPID_EVENT_FILE, index=False)

    print(f"Accounts encoded: {len(features):,}")
    for column in flag_columns:
        print(f"{column}: {int(features[column].sum()):,}")
    print(f"Features saved to: {FEATURE_FILE}")
    print(f"Rapid events saved to: {RAPID_EVENT_FILE}")


if __name__ == "__main__":
    main()
