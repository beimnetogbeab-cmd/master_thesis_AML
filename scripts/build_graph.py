from pathlib import Path

import networkx as nx
import pandas as pd
from pyvis.network import Network


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "gvm_out"
OUTPUT_DIR = PROJECT_DIR / "outputs"

entities = pd.read_csv(
    DATA_DIR / "ground_truth_entities.csv",
    dtype={
        "iban": "string",
        "bic": "string",
        "blz": "string",
    },
)

transactions = pd.read_csv(
    DATA_DIR / "ground_truth_transactions.csv"
)


# Validate identifiers.
if entities["account_id"].isna().any():
    raise ValueError("Some entities have no account_id.")

if entities["account_id"].duplicated().any():
    raise ValueError("Duplicate account_id values found.")

if (
    transactions[
        ["tran_id", "orig_acct", "bene_acct"]
    ]
    .isna()
    .any()
    .any()
):
    raise ValueError(
        "A transaction has a missing ID, sender or recipient."
    )

if transactions["tran_id"].duplicated().any():
    raise ValueError("Duplicate tran_id values found.")

account_ids = set(entities["account_id"])

missing_senders = (
    set(transactions["orig_acct"]) - account_ids
)

missing_recipients = (
    set(transactions["bene_acct"]) - account_ids
)

if missing_senders or missing_recipients:
    raise ValueError(
        f"Unknown accounts in transactions. "
        f"Senders: {missing_senders}; "
        f"recipients: {missing_recipients}"
    )

if (
    transactions[
        ["amount", "timestamp", "tx_type", "is_sar"]
    ]
    .isna()
    .any()
    .any()
):
    raise ValueError(
        "Some transactions have missing attributes."
    )


# MultiDiGraph preserves multiple transactions
# between the same accounts.
graph = nx.MultiDiGraph()


# Add account nodes.
for row in entities.itertuples(index=False):
    graph.add_node(
        int(row.account_id),
        bank_id=str(row.bank_id),
        institution_name=str(row.institution_name),
        account_type=str(row.account_type),
        iban=str(row.iban),
        bic=str(row.bic),
        blz=str(row.blz),
        balance=float(row.balance),
    )


# Add transaction edges.
for row in transactions.itertuples(index=False):
    label_value = str(row.is_sar).strip().lower()

    if label_value not in {
        "true",
        "false",
        "1",
        "0",
    }:
        raise ValueError(
            f"Invalid is_sar value in transaction "
            f"{row.tran_id}"
        )

    graph.add_edge(
        int(row.orig_acct),
        int(row.bene_acct),
        key=int(row.tran_id),
        tran_id=int(row.tran_id),
        amount=float(row.amount),
        timestamp=str(row.timestamp),
        tx_type=str(row.tx_type),
        is_sar=label_value in {"true", "1"},
    )


# Count suspicious transactions.
suspicious_count = sum(
    data["is_sar"]
    for _, _, _, data in graph.edges(
        keys=True,
        data=True,
    )
)


print("Initial directed transaction graph")
print(
    f"Accounts (nodes): "
    f"{graph.number_of_nodes():,}"
)
print(
    f"Transactions (edges): "
    f"{graph.number_of_edges():,}"
)
print(
    f"Suspicious transactions: "
    f"{suspicious_count:,}"
)
print(
    f"Benign transactions: "
    f"{graph.number_of_edges() - suspicious_count:,}"
)


# Create output directory.
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Save the GraphML file.
graphml_file = (
    OUTPUT_DIR
    / "ground_truth_transaction_graph.graphml"
)

nx.write_graphml(
    graph,
    graphml_file,
)

print(f"Graph data saved to: {graphml_file}")


# Generate stable node positions.
positions = nx.spring_layout(
    graph,
    seed=42,
    iterations=150,
)


# Create the interactive network.
view = Network(
    height="850px",
    width="100%",
    directed=True,
    bgcolor="#ffffff",
    font_color="#1f2937",
    select_menu=True,
    filter_menu=True,
    cdn_resources="in_line",
)


bank_colors = {
    "deutsche_bank": "#2563eb",
    "targobank": "#16a34a",
    "commerzbank": "#9333ea",
    "sparkasse": "#ea580c",
}


# Add account nodes to the interactive graph.
for account_id, data in graph.nodes(data=True):
    view.add_node(
        account_id,
        label=(
            f"{data['institution_name']}\n"
            f"Account {account_id}"
        ),
        color=bank_colors.get(
            data["bank_id"],
            "#64748b",
        ),
        size=18,
        x=float(
            positions[account_id][0] * 1800
        ),
        y=float(
            positions[account_id][1] * 1800
        ),
        physics=False,
        details=[
            [
                "Account ID",
                str(account_id),
            ],
            [
                "Bank",
                data["institution_name"],
            ],
            [
                "Account type",
                data["account_type"],
            ],
            [
                "IBAN",
                data["iban"],
            ],
            [
                "BIC",
                data["bic"],
            ],
            [
                "BLZ",
                data["blz"],
            ],
            [
                "Balance",
                f"€{data['balance']:,.2f}",
            ],
        ],
    )


# Add transaction edges to the interactive graph.
for sender, recipient, tran_id, data in graph.edges(
    keys=True,
    data=True,
):
    view.add_edge(
        sender,
        recipient,

        # A unique ID is needed to isolate one transaction.
        id=f"transaction-{tran_id}",

        transaction_id=int(tran_id),
        label=f"€{data['amount']:,.0f}",
        color=(
            "#dc2626"
            if data["is_sar"]
            else "#aab6c2"
        ),
        arrows="to",
        details=[
            [
                "Transaction ID",
                str(tran_id),
            ],
            [
                "From account",
                str(sender),
            ],
            [
                "To account",
                str(recipient),
            ],
            [
                "Amount",
                f"€{data['amount']:,.2f}",
            ],
            [
                "Timestamp",
                data["timestamp"],
            ],
            [
                "Transaction type",
                data["tx_type"],
            ],
            [
                "Suspicious",
                (
                    "Yes"
                    if data["is_sar"]
                    else "No"
                ),
            ],
        ],
    )


view.set_options(
    """
    {
      "physics": {
        "enabled": false
      },
      "edges": {
        "smooth": false,
        "font": {
          "size": 9,
          "strokeWidth": 2,
          "strokeColor": "#ffffff"
        }
      },
      "nodes": {
        "font": {
          "size": 11
        }
      },
      "interaction": {
        "hover": true,
        "navigationButtons": true,
        "multiselect": false
      }
    }
    """
)


# Add the readable information box.
INFO_PANEL_HTML = """
<style>
  #graph-info {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 10000;
    width: min(340px, calc(100vw - 40px));
    max-height: calc(100vh - 40px);
    overflow-y: auto;
    background: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 12px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.18);
    padding: 16px;
    font-family: Arial, sans-serif;
    display: none;
  }

  #graph-info h3 {
    margin: 0 0 12px;
    font-size: 17px;
    color: #111827;
  }

  #graph-info .info-row {
    display: grid;
    grid-template-columns: 110px 1fr;
    gap: 10px;
    padding: 7px 0;
    border-top: 1px solid #eef0f3;
    font-size: 13px;
  }

  #graph-info .info-label {
    color: #64748b;
    font-weight: bold;
  }

  #graph-info .info-value {
    color: #111827;
    overflow-wrap: anywhere;
  }
</style>

<div id="graph-info">
  <h3 id="graph-info-title"></h3>
  <div id="graph-info-rows"></div>
</div>

<script>
  const infoPanel =
    document.getElementById("graph-info");

  const infoTitle =
    document.getElementById("graph-info-title");

  const infoRows =
    document.getElementById("graph-info-rows");

  function showGraphInfo(title, details) {
    infoTitle.textContent = title;
    infoRows.replaceChildren();

    for (const [label, value] of details || []) {
      const row =
        document.createElement("div");

      row.className = "info-row";

      const labelElement =
        document.createElement("div");

      labelElement.className = "info-label";
      labelElement.textContent = label;

      const valueElement =
        document.createElement("div");

      valueElement.className = "info-value";
      valueElement.textContent = value;

      row.append(
        labelElement,
        valueElement,
      );

      infoRows.appendChild(row);
    }

    infoPanel.style.display = "block";
  }

  function hideGraphInfo() {
    infoPanel.style.display = "none";
  }

  network.on("hoverNode", function (event) {
    const node = nodes.get(event.node);

    if (node) {
      showGraphInfo(
        "Bank account",
        node.details,
      );
    }
  });

  network.on("blurNode", hideGraphInfo);

  network.on("hoverEdge", function (event) {
    const edge = edges.get(event.edge);

    if (edge) {
      showGraphInfo(
        "Transaction",
        edge.details,
      );
    }
  });

  network.on("blurEdge", hideGraphInfo);
</script>
"""


# Add single-item isolation without replacing
# the existing PyVis menus.
SINGLE_SELECTION_HTML = """
<style>
  #selection-controls {
    position: fixed;
    left: 20px;
    bottom: 20px;
    z-index: 10000;
    padding: 12px;
    background: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 10px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.16);
    font-family: Arial, sans-serif;
  }

  #selection-controls button {
    display: block;
    width: 190px;
    margin: 5px 0;
    padding: 8px;
    cursor: pointer;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    background: #f8fafc;
    color: #1f2937;
  }

  #selection-controls button:hover {
    background: #e2e8f0;
  }

  #selection-message {
    width: 190px;
    margin-top: 7px;
    font-size: 12px;
    line-height: 1.4;
    color: #475569;
  }
</style>

<div id="selection-controls">
  <button id="show-selected-only">
    Show selected item only
  </button>

  <button id="restore-full-graph">
    Show full graph
  </button>

  <div id="selection-message">
    Select one account or transaction.
  </div>
</div>

<script>
  /*
   * Save copies of the complete graph.
   */
  const completeNodes = nodes
    .get()
    .map(node => ({ ...node }));

  const completeEdges = edges
    .get()
    .map(edge => ({ ...edge }));

  const selectionMessage =
    document.getElementById(
      "selection-message"
    );

  let selectedNodeId = null;
  let selectedEdgeId = null;


  /*
   * Remember the account clicked by the user.
   */
  network.on("selectNode", function (event) {
    if (event.nodes.length === 1) {
      selectedNodeId = event.nodes[0];
      selectedEdgeId = null;

      selectionMessage.textContent =
        `Selected account: ${selectedNodeId}`;
    }
  });


  /*
   * Remember the transaction clicked by the user.
   */
  network.on("selectEdge", function (event) {
    if (
      event.nodes.length === 0 &&
      event.edges.length === 1
    ) {
      selectedEdgeId = event.edges[0];
      selectedNodeId = null;

      const selectedEdge = completeEdges.find(
        edge =>
          String(edge.id) ===
          String(selectedEdgeId)
      );

      if (selectedEdge) {
        selectionMessage.textContent =
          `Selected transaction: ` +
          `${selectedEdge.transaction_id}`;
      }
    }
  });


  /*
   * Show only the selected account or transaction.
   */
  document
    .getElementById("show-selected-only")
    .addEventListener("click", function () {
      /*
       * Show one account and no transactions.
       */
      if (selectedNodeId !== null) {
        const selectedNode = completeNodes.find(
          node =>
            String(node.id) ===
            String(selectedNodeId)
        );

        if (!selectedNode) {
          return;
        }

        network.unselectAll();
        edges.clear();
        nodes.clear();

        nodes.add({ ...selectedNode });

        selectionMessage.textContent =
          `Showing only account ${selectedNodeId}`;

        hideGraphInfo();

        window.setTimeout(function () {
          network.fit({
            animation: false,
          });
        }, 50);

        return;
      }


      /*
       * A transaction requires its sender and recipient
       * nodes to remain visible.
       */
      if (selectedEdgeId !== null) {
        const selectedEdge = completeEdges.find(
          edge =>
            String(edge.id) ===
            String(selectedEdgeId)
        );

        if (!selectedEdge) {
          return;
        }

        const endpointNodes = completeNodes.filter(
          node =>
            String(node.id) ===
              String(selectedEdge.from) ||
            String(node.id) ===
              String(selectedEdge.to)
        );

        network.unselectAll();
        edges.clear();
        nodes.clear();

        nodes.add(
          endpointNodes.map(
            node => ({ ...node })
          )
        );

        edges.add({ ...selectedEdge });

        selectionMessage.textContent =
          `Showing only transaction ` +
          `${selectedEdge.transaction_id}`;

        hideGraphInfo();

        window.setTimeout(function () {
          network.fit({
            animation: false,
          });
        }, 50);

        return;
      }

      window.alert(
        "Please click one account or transaction first."
      );
    });


  /*
   * Restore the complete graph.
   */
  document
    .getElementById("restore-full-graph")
    .addEventListener("click", function () {
      network.unselectAll();
      edges.clear();
      nodes.clear();

      nodes.add(
        completeNodes.map(
          node => ({ ...node })
        )
      );

      edges.add(
        completeEdges.map(
          edge => ({ ...edge })
        )
      );

      selectedNodeId = null;
      selectedEdgeId = null;

      selectionMessage.textContent =
        "Full graph displayed.";

      hideGraphInfo();

      window.setTimeout(function () {
        network.fit({
          animation: false,
        });
      }, 50);
    });
</script>
"""


# Generate the interactive HTML.
html_file = (
    OUTPUT_DIR
    / "interactive_transaction_graph.html"
)

html = view.generate_html(
    notebook=False,
)

html = html.replace(
    "</body>",
    (
        INFO_PANEL_HTML
        + SINGLE_SELECTION_HTML
        + "\n</body>"
    ),
)

html_file.write_text(
    html,
    encoding="utf-8",
)

print(f"Interactive graph saved to: {html_file}")