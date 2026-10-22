from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import pandas as pd
from pyvis.network import Network


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
DEFAULT_INPUT = PROJECT_DIR / "IBM_PUBLIC_DATASET" / "HI-Small_Trans.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "ibm_aml"

GRAPH_FILE = OUTPUT_DIR / "ibm_transaction_graph.graphml"
SUMMARY_FILE = OUTPUT_DIR / "ibm_dataset_summary.txt"
HTML_FILE = OUTPUT_DIR / "ibm_graph_sample.html"

MAX_HTML_EDGES = 1_000
MAX_HTML_SUSPICIOUS_EDGES = 500


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a directed transaction graph from IBM AML data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to HI_Small_Trans.csv",
    )
    parser.add_argument(
        "--skip-html",
        action="store_true",
        help="Build the graph without creating the HTML sample.",
    )
    parser.add_argument(
        "--skip-graphml",
        action="store_true",
        help="Run validation/analysis without writing the large GraphML file.",
    )
    return parser.parse_args()


def load_transactions(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"IBM transaction file not found: {file_path}")

    transactions = pd.read_csv(
        file_path,
        dtype={
            "From Bank": "string",
            "Account": "string",
            "To Bank": "string",
            "Account.1": "string",
            "Receiving Currency": "string",
            "Payment Currency": "string",
            "Payment Format": "string",
        },
    )

    # Pandas renames the second duplicate "Account" header to "Account.1".
    required_columns = {
        "Timestamp",
        "From Bank",
        "Account",
        "To Bank",
        "Account.1",
        "Amount Received",
        "Receiving Currency",
        "Amount Paid",
        "Payment Currency",
        "Payment Format",
        "Is Laundering",
    }
    missing_columns = required_columns - set(transactions.columns)
    if missing_columns:
        raise ValueError(
            "The IBM file is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    transactions = transactions.rename(
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

    required_values = [
        "timestamp",
        "from_bank",
        "from_account",
        "to_bank",
        "to_account",
        "amount_received",
        "amount_paid",
        "is_laundering",
    ]
    if transactions[required_values].isna().any().any():
        missing = transactions[required_values].isna().sum()
        missing = missing[missing > 0].to_dict()
        raise ValueError(f"Required transaction values are missing: {missing}")

    transactions["timestamp"] = pd.to_datetime(
        transactions["timestamp"], errors="raise"
    )
    transactions["amount_received"] = pd.to_numeric(
        transactions["amount_received"], errors="raise"
    )
    transactions["amount_paid"] = pd.to_numeric(
        transactions["amount_paid"], errors="raise"
    )
    transactions["is_laundering"] = pd.to_numeric(
        transactions["is_laundering"], errors="raise"
    ).astype("int8")

    invalid_labels = set(transactions["is_laundering"].unique()) - {0, 1}
    if invalid_labels:
        raise ValueError(f"Invalid Is Laundering labels: {invalid_labels}")

    # Bank is included to prevent accounts at different banks from colliding.
    transactions["sender_id"] = (
        transactions["from_bank"].astype(str)
        + "::"
        + transactions["from_account"].astype(str)
    )
    transactions["recipient_id"] = (
        transactions["to_bank"].astype(str)
        + "::"
        + transactions["to_account"].astype(str)
    )
    transactions.insert(0, "transaction_id", range(1, len(transactions) + 1))

    return transactions


def create_summary(transactions: pd.DataFrame) -> str:
    account_ids = pd.concat(
        [transactions["sender_id"], transactions["recipient_id"]],
        ignore_index=True,
    )
    suspicious = int(transactions["is_laundering"].sum())
    benign = len(transactions) - suspicious
    suspicious_rate = suspicious / len(transactions) if len(transactions) else 0

    lines = [
        "IBM AML dataset summary",
        f"Transactions: {len(transactions):,}",
        f"Unique accounts: {account_ids.nunique():,}",
        f"Unique banks: {pd.concat([transactions['from_bank'], transactions['to_bank']]).nunique():,}",
        f"Laundering transactions: {suspicious:,}",
        f"Normal transactions: {benign:,}",
        f"Laundering rate: {suspicious_rate:.4%}",
        f"Duplicate complete rows: {transactions.drop(columns=['transaction_id']).duplicated().sum():,}",
        f"Start timestamp: {transactions['timestamp'].min()}",
        f"End timestamp: {transactions['timestamp'].max()}",
        f"Payment formats: {transactions['payment_format'].nunique():,}",
        f"Payment currencies: {transactions['payment_currency'].nunique():,}",
        f"Receiving currencies: {transactions['receiving_currency'].nunique():,}",
    ]
    return "\n".join(lines)


def build_graph(transactions: pd.DataFrame) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()

    sender_nodes = transactions[["sender_id", "from_bank", "from_account"]].rename(
        columns={
            "sender_id": "node_id",
            "from_bank": "bank_id",
            "from_account": "account_id",
        }
    )
    recipient_nodes = transactions[
        ["recipient_id", "to_bank", "to_account"]
    ].rename(
        columns={
            "recipient_id": "node_id",
            "to_bank": "bank_id",
            "to_account": "account_id",
        }
    )
    nodes = pd.concat([sender_nodes, recipient_nodes], ignore_index=True)
    nodes = nodes.drop_duplicates(subset="node_id")

    for row in nodes.itertuples(index=False):
        graph.add_node(
            str(row.node_id),
            bank_id=str(row.bank_id),
            account_id=str(row.account_id),
        )

    for row in transactions.itertuples(index=False):
        graph.add_edge(
            str(row.sender_id),
            str(row.recipient_id),
            key=int(row.transaction_id),
            transaction_id=int(row.transaction_id),
            timestamp=row.timestamp.isoformat(),
            amount_paid=float(row.amount_paid),
            payment_currency=str(row.payment_currency),
            amount_received=float(row.amount_received),
            receiving_currency=str(row.receiving_currency),
            payment_format=str(row.payment_format),
            is_laundering=bool(row.is_laundering),
        )

    return graph


def select_html_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    suspicious = transactions[transactions["is_laundering"] == 1].head(
        MAX_HTML_SUSPICIOUS_EDGES
    )
    remaining = MAX_HTML_EDGES - len(suspicious)
    benign = transactions[transactions["is_laundering"] == 0].head(remaining)
    return pd.concat([suspicious, benign], ignore_index=True)


def create_html_sample(transactions: pd.DataFrame) -> None:
    selected = select_html_transactions(transactions)
    sample_graph = build_graph(selected)
    positions = nx.spring_layout(sample_graph, seed=42, iterations=100)

    view = Network(
        height="850px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#1f2937",
        cdn_resources="in_line",
    )

    for node_id, data in sample_graph.nodes(data=True):
        view.add_node(
            node_id,
            label=f"Bank {data['bank_id']}\n{data['account_id']}",
            account_id=str(data["account_id"]),
            bank_id=str(data["bank_id"]),
            color="#2563eb",
            size=16,
            x=float(positions[node_id][0] * 1_500),
            y=float(positions[node_id][1] * 1_500),
            physics=False,
            details=[
                ["Account", str(data["account_id"])],
                ["Bank", str(data["bank_id"])],
                ["Graph node ID", str(node_id)],
            ],
        )

    for sender, recipient, key, data in sample_graph.edges(keys=True, data=True):
        view.add_edge(
            sender,
            recipient,
            id=f"transaction-{key}",
            transaction_id=int(data["transaction_id"]),
            color="#dc2626" if data["is_laundering"] else "#cbd5e1",
            arrows="to",
            details=[
                ["Transaction ID", str(data["transaction_id"])],
                ["From", str(sender)],
                ["To", str(recipient)],
                ["Timestamp", str(data["timestamp"])],
                [
                    "Amount paid",
                    f"{data['amount_paid']:,.2f} {data['payment_currency']}",
                ],
                [
                    "Amount received",
                    f"{data['amount_received']:,.2f} {data['receiving_currency']}",
                ],
                ["Payment format", str(data["payment_format"])],
                [
                    "Laundering",
                    "Yes" if data["is_laundering"] else "No",
                ],
            ],
        )

    view.set_options(
        """
        {
          "physics": {"enabled": false},
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "multiselect": false
          },
          "edges": {"smooth": false}
        }
        """
    )

    custom_html = """
    <style>
      #graph-search, #graph-info {
        position: fixed;
        z-index: 10000;
        background: #ffffff;
        border: 1px solid #d1d5db;
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.16);
        font-family: Arial, sans-serif;
      }
      #graph-search {
        top: 20px;
        left: 20px;
        width: min(390px, calc(100vw - 40px));
        padding: 14px;
      }
      #graph-search h3, #graph-info h3 {
        margin: 0 0 10px;
        color: #111827;
        font-size: 16px;
      }
      #search-row {
        display: grid;
        grid-template-columns: 110px 1fr auto;
        gap: 8px;
      }
      #search-row select, #search-row input, #search-row button,
      #clear-search {
        border: 1px solid #cbd5e1;
        border-radius: 7px;
        padding: 8px;
        font-size: 13px;
      }
      #search-row button {
        border-color: #2563eb;
        background: #2563eb;
        color: white;
        cursor: pointer;
      }
      #clear-search {
        margin-top: 8px;
        background: #f8fafc;
        cursor: pointer;
      }
      #search-status {
        margin-top: 8px;
        min-height: 16px;
        color: #64748b;
        font-size: 12px;
      }
      #graph-info {
        top: 20px;
        right: 20px;
        width: min(360px, calc(100vw - 40px));
        max-height: calc(100vh - 40px);
        overflow-y: auto;
        padding: 16px;
        display: none;
      }
      .info-row {
        display: grid;
        grid-template-columns: 120px 1fr;
        gap: 10px;
        padding: 7px 0;
        border-top: 1px solid #eef0f3;
        font-size: 13px;
      }
      .info-label {color: #64748b; font-weight: bold;}
      .info-value {color: #111827; overflow-wrap: anywhere;}
    </style>

    <div id="graph-search">
      <h3>Search displayed graph</h3>
      <div id="search-row">
        <select id="search-type">
          <option value="node">Account</option>
          <option value="edge">Transaction</option>
        </select>
        <input id="search-value" placeholder="Enter account or ID">
        <button id="search-button" type="button">Search</button>
      </div>
      <button id="clear-search" type="button">Clear selection</button>
      <div id="search-status"></div>
    </div>

    <div id="graph-info">
      <h3 id="graph-info-title"></h3>
      <div id="graph-info-rows"></div>
    </div>

    <script>
      const infoPanel = document.getElementById("graph-info");
      const infoTitle = document.getElementById("graph-info-title");
      const infoRows = document.getElementById("graph-info-rows");
      const searchStatus = document.getElementById("search-status");

      function showGraphInfo(title, details) {
        infoTitle.textContent = title;
        infoRows.replaceChildren();
        for (const [label, value] of details || []) {
          const row = document.createElement("div");
          row.className = "info-row";
          const labelElement = document.createElement("div");
          labelElement.className = "info-label";
          labelElement.textContent = label;
          const valueElement = document.createElement("div");
          valueElement.className = "info-value";
          valueElement.textContent = value;
          row.append(labelElement, valueElement);
          infoRows.appendChild(row);
        }
        infoPanel.style.display = "block";
      }

      function hideGraphInfo() {
        infoPanel.style.display = "none";
      }

      network.on("hoverNode", function (event) {
        const node = nodes.get(event.node);
        showGraphInfo("Bank account", node.details);
      });
      network.on("blurNode", hideGraphInfo);
      network.on("hoverEdge", function (event) {
        const edge = edges.get(event.edge);
        showGraphInfo("Transaction", edge.details);
      });
      network.on("blurEdge", hideGraphInfo);

      function searchGraph() {
        const type = document.getElementById("search-type").value;
        const query = document.getElementById("search-value").value.trim().toLowerCase();
        if (!query) {
          searchStatus.textContent = "Enter a search value.";
          return;
        }

        if (type === "node") {
          const matches = nodes.get({filter: function (node) {
            return String(node.id).toLowerCase() === query ||
              String(node.account_id || "").toLowerCase() === query ||
              String(node.label || "").toLowerCase().includes(query);
          }});
          if (!matches.length) {
            searchStatus.textContent = "Account not found in the displayed sample.";
            return;
          }
          const node = matches[0];
          network.selectNodes([node.id]);
          network.focus(node.id, {scale: 1.4, animation: true});
          showGraphInfo("Bank account", node.details);
          searchStatus.textContent = matches.length > 1
            ? `Found ${matches.length} accounts; showing the first.`
            : "Account found.";
        } else {
          const matches = edges.get({filter: function (edge) {
            return String(edge.transaction_id) === query ||
              String(edge.id).toLowerCase() === query;
          }});
          if (!matches.length) {
            searchStatus.textContent = "Transaction not found in the displayed sample.";
            return;
          }
          const edge = matches[0];
          network.selectEdges([edge.id]);
          network.fit({nodes: [edge.from, edge.to], animation: true});
          showGraphInfo("Transaction", edge.details);
          searchStatus.textContent = "Transaction found.";
        }
      }

      document.getElementById("search-button").addEventListener("click", searchGraph);
      document.getElementById("search-value").addEventListener("keydown", function (event) {
        if (event.key === "Enter") searchGraph();
      });
      document.getElementById("clear-search").addEventListener("click", function () {
        network.unselectAll();
        network.fit({animation: true});
        document.getElementById("search-value").value = "";
        searchStatus.textContent = "";
        hideGraphInfo();
      });
    </script>
    """

    html = view.generate_html(notebook=False)
    html = html.replace("</body>", custom_html + "\n</body>")
    HTML_FILE.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_arguments()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    transactions = load_transactions(args.input)
    summary = create_summary(transactions)
    print(summary)
    SUMMARY_FILE.write_text(summary + "\n", encoding="utf-8")
    print(f"\nSummary saved to: {SUMMARY_FILE}")

    if not args.skip_graphml:
        graph = build_graph(transactions)
        print(f"Graph nodes: {graph.number_of_nodes():,}")
        print(f"Graph edges: {graph.number_of_edges():,}")
        nx.write_graphml(graph, GRAPH_FILE)
        print(f"Full graph saved to: {GRAPH_FILE}")
    else:
        print("Full GraphML creation skipped.")

    if not args.skip_html:
        create_html_sample(transactions)
        print(f"HTML sample saved to: {HTML_FILE}")


if __name__ == "__main__":
    main()
