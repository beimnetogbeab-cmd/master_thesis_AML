from bisect import bisect_left

from pathlib import Path



import networkx as nx

import pandas as pd





PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data" / "gvm_out"

OUTPUT_DIR = PROJECT_DIR / "outputs"



ENTITY_FILE = DATA_DIR / "ground_truth_entities.csv"

TRANSACTION_FILE = DATA_DIR / "ground_truth_transactions.csv"

GRAPH_FILE = OUTPUT_DIR / "ground_truth_transaction_graph.graphml"



FEATURE_FILE = OUTPUT_DIR / "account_typology_features.csv"

RAPID_EVENT_FILE = OUTPUT_DIR / "rapid_pass_through_events.csv"

ENRICHED_GRAPH_FILE = (

    OUTPUT_DIR

    / "transaction_graph_with_typology_features.graphml"

)





STRUCTURING_THRESHOLD = 10_000

STRUCTURING_LOWER_BOUND = 1_000

STRUCTURING_MIN_TRANSACTIONS = 3

FAN_IN_MIN_SENDERS = 5

FAN_OUT_MIN_RECIPIENTS = 5

RAPID_PASS_THROUGH_HOURS = 24

RAPID_AMOUNT_MIN_RATIO = 0.80

RAPID_AMOUNT_MAX_RATIO = 1.20









entities = pd.read_csv(ENTITY_FILE)



transactions = pd.read_csv(

    TRANSACTION_FILE,

    parse_dates=["timestamp"],

)



transactions["timestamp"] = pd.to_datetime(

    transactions["timestamp"],

    utc=True,

)





# ---------------------------------------------------------

# CREATE EMPTY ACCOUNT FEATURE TABLE

# ---------------------------------------------------------



features = pd.DataFrame(

    index=entities["account_id"].astype(int)

)



features.index.name = "account_id"





# ---------------------------------------------------------

# BASIC INCOMING FEATURES

# ---------------------------------------------------------



incoming = transactions.groupby("bene_acct").agg(

    incoming_transaction_count=("tran_id", "count"),

    unique_senders=("orig_acct", "nunique"),

    total_incoming_amount=("amount", "sum"),

    average_incoming_amount=("amount", "mean"),

    maximum_incoming_amount=("amount", "max"),

)



features = features.join(

    incoming,

    how="left",

)





# ---------------------------------------------------------

# BASIC OUTGOING FEATURES

# ---------------------------------------------------------



outgoing = transactions.groupby("orig_acct").agg(

    outgoing_transaction_count=("tran_id", "count"),

    unique_recipients=("bene_acct", "nunique"),

    total_outgoing_amount=("amount", "sum"),

    average_outgoing_amount=("amount", "mean"),

    maximum_outgoing_amount=("amount", "max"),

)



features = features.join(

    outgoing,

    how="left",

)





# Replace missing aggregate values with zero.

numeric_columns = features.columns



features[numeric_columns] = (

    features[numeric_columns]

    .fillna(0)

)





# ---------------------------------------------------------

# FAN-IN FEATURES

# ---------------------------------------------------------



# Fan-in means that several different accounts send money

# to one receiving account.

features["fan_in_score"] = (

    features["unique_senders"]

)



features["fan_in_flag"] = (

    features["unique_senders"]

    >= FAN_IN_MIN_SENDERS

).astype(int)





# ---------------------------------------------------------

# FAN-OUT FEATURES

# ---------------------------------------------------------



# Fan-out means that one account distributes money

# to several different recipient accounts.

features["fan_out_score"] = (

    features["unique_recipients"]

)



features["fan_out_flag"] = (

    features["unique_recipients"]

    >= FAN_OUT_MIN_RECIPIENTS

).astype(int)





# ---------------------------------------------------------

# STRUCTURING FEATURES

# ---------------------------------------------------------



# Structuring is approximated as repeated transactions

# immediately below an assumed threshold.

near_threshold = transactions[

    (

        transactions["amount"]

        >= STRUCTURING_LOWER_BOUND

    )

    & (

        transactions["amount"]

        < STRUCTURING_THRESHOLD

    )

].copy()



near_threshold["transaction_day"] = (

    near_threshold["timestamp"].dt.floor("D")

)



structuring_windows = (

    near_threshold

    .groupby(

        ["orig_acct", "transaction_day"]

    )

    .agg(

        near_threshold_transaction_count=(

            "tran_id",

            "count",

        ),

        near_threshold_total_amount=(

            "amount",

            "sum",

        ),

    )

    .reset_index()

)



structuring_windows["is_structuring_window"] = (

    structuring_windows[

        "near_threshold_transaction_count"

    ]

    >= STRUCTURING_MIN_TRANSACTIONS

)



structuring_summary = (

    structuring_windows

    .groupby("orig_acct")

    .agg(

        near_threshold_transaction_count=(

            "near_threshold_transaction_count",

            "sum",

        ),

        maximum_near_threshold_daily_count=(

            "near_threshold_transaction_count",

            "max",

        ),

        structuring_window_count=(

            "is_structuring_window",

            "sum",

        ),

    )

)



features = features.join(

    structuring_summary,

    how="left",

)



structuring_columns = [

    "near_threshold_transaction_count",

    "maximum_near_threshold_daily_count",

    "structuring_window_count",

]



features[structuring_columns] = (

    features[structuring_columns]

    .fillna(0)

)



features["structuring_flag"] = (

    features["structuring_window_count"] > 0

).astype(int)





# ---------------------------------------------------------

# CIRCULAR TRANSFER FEATURES

# ---------------------------------------------------------



# Create a simple directed graph for identifying cycles.

simple_graph = nx.DiGraph()



simple_graph.add_nodes_from(

    entities["account_id"].astype(int)

)



simple_graph.add_edges_from(

    transactions[

        ["orig_acct", "bene_acct"]

    ].itertuples(

        index=False,

        name=None,

    )

)



adjacency = {

    node: set(simple_graph.successors(node))

    for node in simple_graph.nodes()

}



two_node_cycles = set()

three_node_cycles = set()



for first_account in simple_graph.nodes():

    for second_account in adjacency[first_account]:



        # A -> B and B -> A

        if first_account in adjacency.get(

            second_account,

            set(),

        ):

            cycle = tuple(

                sorted(

                    [

                        first_account,

                        second_account,

                    ]

                )

            )



            two_node_cycles.add(cycle)



        # A -> B -> C -> A

        for third_account in adjacency.get(

            second_account,

            set(),

        ):

            if third_account in {

                first_account,

                second_account,

            }:

                continue



            if first_account in adjacency.get(

                third_account,

                set(),

            ):

                cycle = tuple(

                    sorted(

                        [

                            first_account,

                            second_account,

                            third_account,

                        ]

                    )

                )



                three_node_cycles.add(cycle)





two_cycle_counts = {

    account_id: 0

    for account_id in features.index

}



three_cycle_counts = {

    account_id: 0

    for account_id in features.index

}



for cycle in two_node_cycles:

    for account_id in cycle:

        two_cycle_counts[account_id] += 1



for cycle in three_node_cycles:

    for account_id in cycle:

        three_cycle_counts[account_id] += 1





features["two_account_cycle_count"] = (

    pd.Series(two_cycle_counts)

)



features["three_account_cycle_count"] = (

    pd.Series(three_cycle_counts)

)



features[

    [

        "two_account_cycle_count",

        "three_account_cycle_count",

    ]

] = features[

    [

        "two_account_cycle_count",

        "three_account_cycle_count",

    ]

].fillna(0)



features["circular_transfer_count"] = (

    features["two_account_cycle_count"]

    + features["three_account_cycle_count"]

)



features["two_way_circular_flag"] = (

    features["two_account_cycle_count"] > 0

).astype(int)



features["three_way_circular_flag"] = (

    features["three_account_cycle_count"] > 0

).astype(int)



features["circular_transfer_flag"] = (

    features["circular_transfer_count"] > 0

).astype(int)





# ---------------------------------------------------------

# RAPID PASS-THROUGH FEATURES

# ---------------------------------------------------------



rapid_counts = {

    account_id: 0

    for account_id in features.index

}



rapid_delay_sum = {

    account_id: 0.0

    for account_id in features.index

}



# One row will be saved for every matched incoming/outgoing
# rapid pass-through pair.
rapid_events = []



maximum_delay_seconds = (

    RAPID_PASS_THROUGH_HOURS * 60 * 60

)



for account_id in features.index:

    incoming_account_transactions = (

        transactions[

            transactions["bene_acct"]

            == account_id

        ]

        .sort_values("timestamp")

    )



    outgoing_account_transactions = (

        transactions[

            transactions["orig_acct"]

            == account_id

        ]

        .sort_values("timestamp")

    )



    if (

        incoming_account_transactions.empty

        or outgoing_account_transactions.empty

    ):

        continue



    outgoing_records = list(

        outgoing_account_transactions[

            ["tran_id", "timestamp", "amount"]

        ].itertuples(

            index=False,

            name=None,

        )

    )



    outgoing_times = [

        record[1]

        for record in outgoing_records

    ]



    for incoming_row in (

        incoming_account_transactions

        .itertuples(index=False)

    ):

        incoming_transaction_id = int(incoming_row.tran_id)

        incoming_time = incoming_row.timestamp

        incoming_amount = float(

            incoming_row.amount

        )



        # Find the first outgoing transaction that occurs

        # at or after the incoming transaction.

        position = bisect_left(

            outgoing_times,

            incoming_time,

        )



        while position < len(outgoing_records):

            (
                outgoing_transaction_id,
                outgoing_time,
                outgoing_amount,
            ) = (

                outgoing_records[position]

            )



            delay_seconds = (

                outgoing_time - incoming_time

            ).total_seconds()



            if delay_seconds > maximum_delay_seconds:

                break



            if delay_seconds >= 0:

                amount_ratio = (

                    float(outgoing_amount)

                    / incoming_amount

                    if incoming_amount > 0

                    else 0

                )



                if (

                    RAPID_AMOUNT_MIN_RATIO

                    <= amount_ratio

                    <= RAPID_AMOUNT_MAX_RATIO

                ):

                    rapid_counts[account_id] += 1



                    rapid_delay_sum[account_id] += (

                        delay_seconds / 3600

                    )



                    rapid_events.append(

                        {

                            "account_id": int(account_id),

                            "incoming_tran_id": incoming_transaction_id,

                            "incoming_timestamp": incoming_time.isoformat(),

                            "incoming_amount": incoming_amount,

                            "outgoing_tran_id": int(outgoing_transaction_id),

                            "outgoing_timestamp": outgoing_time.isoformat(),

                            "outgoing_amount": float(outgoing_amount),

                            "delay_hours": delay_seconds / 3600,

                            "amount_ratio": amount_ratio,

                        }

                    )



                    # Count only the first matching outgoing

                    # transaction for this incoming transaction.

                    break



            position += 1





features["rapid_pass_through_count"] = (

    pd.Series(rapid_counts)

)



features["rapid_pass_through_flag"] = (

    features["rapid_pass_through_count"] > 0

).astype(int)





def calculate_average_delay(account_id):

    count = rapid_counts[account_id]



    if count == 0:

        return 0.0



    return rapid_delay_sum[account_id] / count





features[

    "average_rapid_pass_through_hours"

] = [

    calculate_average_delay(account_id)

    for account_id in features.index

]



# Keep the first matched pair as compact node features for
# GraphML and the interactive visualization. All matched pairs
# are also written separately to rapid_pass_through_events.csv.
rapid_event_columns = [

    "account_id",

    "incoming_tran_id",

    "incoming_timestamp",

    "incoming_amount",

    "outgoing_tran_id",

    "outgoing_timestamp",

    "outgoing_amount",

    "delay_hours",

    "amount_ratio",

]



rapid_events_df = pd.DataFrame(

    rapid_events,

    columns=rapid_event_columns,

)



if rapid_events_df.empty:

    first_rapid_events = pd.DataFrame(

        index=features.index,

        columns=[

            "rapid_incoming_timestamp",

            "rapid_outgoing_timestamp",

            "rapid_example_delay_hours",

        ],

    )

else:

    first_rapid_events = (

        rapid_events_df

        .sort_values("incoming_timestamp")

        .drop_duplicates("account_id", keep="first")

        .set_index("account_id")

        [[

            "incoming_timestamp",

            "outgoing_timestamp",

            "delay_hours",

        ]]

        .rename(

            columns={

                "incoming_timestamp": "rapid_incoming_timestamp",

                "outgoing_timestamp": "rapid_outgoing_timestamp",

                "delay_hours": "rapid_example_delay_hours",

            }

        )

    )



features = features.join(

    first_rapid_events,

    how="left",

)



features[

    [

        "rapid_incoming_timestamp",

        "rapid_outgoing_timestamp",

    ]

] = features[

    [

        "rapid_incoming_timestamp",

        "rapid_outgoing_timestamp",

    ]

].fillna("")



features["rapid_example_delay_hours"] = (

    features["rapid_example_delay_hours"].fillna(0.0)

)





# ---------------------------------------------------------

# COMBINED TYPOLOGY FEATURE

# ---------------------------------------------------------



flag_columns = [

    "structuring_flag",

    "fan_in_flag",

    "fan_out_flag",

    "circular_transfer_flag",

    "rapid_pass_through_flag",

]



features["typology_flag_count"] = (

    features[flag_columns].sum(axis=1)

)



features["has_typology_indicator"] = (

    features["typology_flag_count"] > 0

).astype(int)





# Convert count columns to integers.

count_columns = [

    "incoming_transaction_count",

    "unique_senders",

    "outgoing_transaction_count",

    "unique_recipients",

    "fan_in_score",

    "fan_in_flag",

    "fan_out_score",

    "fan_out_flag",

    "near_threshold_transaction_count",

    "maximum_near_threshold_daily_count",

    "structuring_window_count",

    "structuring_flag",

    "two_account_cycle_count",

    "three_account_cycle_count",

    "two_way_circular_flag",

    "three_way_circular_flag",

    "circular_transfer_count",

    "circular_transfer_flag",

    "rapid_pass_through_count",

    "rapid_pass_through_flag",

    "typology_flag_count",

    "has_typology_indicator",

]



for column in count_columns:

    features[column] = (

        features[column].astype(int)

    )





# ---------------------------------------------------------

# SAVE FEATURE TABLE

# ---------------------------------------------------------



OUTPUT_DIR.mkdir(

    parents=True,

    exist_ok=True,

)



features.reset_index().to_csv(

    FEATURE_FILE,

    index=False,

)



rapid_events_df.to_csv(

    RAPID_EVENT_FILE,

    index=False,

)



print(f"Feature table saved to: {FEATURE_FILE}")

print(f"Rapid pass-through events saved to: {RAPID_EVENT_FILE}")





# ---------------------------------------------------------

# ADD FEATURES TO THE EXISTING GRAPHML FILE

# ---------------------------------------------------------



if not GRAPH_FILE.exists():

    raise FileNotFoundError(

        f"Graph file not found: {GRAPH_FILE}. "

        "Run scripts/build_graph.py first."

    )



enriched_graph = nx.read_graphml(

    GRAPH_FILE

)



feature_records = (

    features.reset_index()

    .set_index("account_id")

    .to_dict(orient="index")

)



for graph_node in enriched_graph.nodes():

    account_id = int(graph_node)



    if account_id not in feature_records:

        continue



    for feature_name, feature_value in (

        feature_records[account_id].items()

    ):

        if pd.isna(feature_value):

            feature_value = 0



        enriched_graph.nodes[graph_node][

            feature_name

        ] = feature_value





nx.write_graphml(

    enriched_graph,

    ENRICHED_GRAPH_FILE,

)



print(

    "Enriched graph saved to: "

    f"{ENRICHED_GRAPH_FILE}"

)





# ---------------------------------------------------------

# PRINT SUMMARY

# ---------------------------------------------------------



print("\nAML typology feature summary")



for flag_column in flag_columns:

    print(

        f"{flag_column}: "

        f"{int(features[flag_column].sum()):,} "

        "accounts"

    )



print(

    "Accounts with at least one indicator: "

    f"{int(features['has_typology_indicator'].sum()):,}"

)
