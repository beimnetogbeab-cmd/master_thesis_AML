from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd
from pyvis.network import Network


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
DATA_FILE = PROJECT_DIR / "IBM_PUBLIC_DATASET" / "HI-Small_Trans.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "ibm_aml"
FEATURE_FILE = OUTPUT_DIR / "ibm_account_features.csv"
HTML_FILE = OUTPUT_DIR / "ibm_typology_graph.html"

MAX_ACCOUNTS = 150
MAX_EDGES = 1_000

FLAG_LABELS = {
    "structuring_flag": "Structuring",
    "fan_in_flag": "Fan-in",
    "fan_out_flag": "Fan-out",
    "two_way_circular_flag": "Two-way circular (A-B-A)",
    "three_way_circular_flag": "Three-way circular (A-B-C-A)",
    "rapid_pass_through_flag": "Rapid pass-through",
}


def load_visualization_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"IBM transaction file not found: {DATA_FILE}")
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURE_FILE}. Run encode_ibm_typology_features.py first."
        )

    features = pd.read_csv(FEATURE_FILE)
    ranked = features.sort_values(
        ["typology_flag_count", "is_laundering_account"], ascending=False
    )
    selected_accounts = set(ranked.head(MAX_ACCOUNTS)["account_key"].astype(str))

    transactions = pd.read_csv(
        DATA_FILE,
        usecols=[
            "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
            "Amount Received", "Receiving Currency", "Amount Paid",
            "Payment Currency", "Payment Format", "Is Laundering",
        ],
        dtype={
            "From Bank": "string", "Account": "string",
            "To Bank": "string", "Account.1": "string",
            "Is Laundering": "int8",
        },
    ).rename(
        columns={
            "Timestamp": "timestamp", "From Bank": "from_bank",
            "Account": "from_account", "To Bank": "to_bank",
            "Account.1": "to_account", "Amount Received": "amount_received",
            "Receiving Currency": "receiving_currency", "Amount Paid": "amount_paid",
            "Payment Currency": "payment_currency", "Payment Format": "payment_format",
            "Is Laundering": "is_laundering",
        }
    )
    transactions["transaction_id"] = transactions.index + 1
    transactions["sender_id"] = transactions["from_bank"].astype(str) + "::" + transactions["from_account"].astype(str)
    transactions["recipient_id"] = transactions["to_bank"].astype(str) + "::" + transactions["to_account"].astype(str)

    related = transactions[
        transactions["sender_id"].isin(selected_accounts)
        | transactions["recipient_id"].isin(selected_accounts)
    ]
    laundering = related[related["is_laundering"] == 1].head(MAX_EDGES // 2)
    normal = related[
        (related["is_laundering"] == 0)
        & ~related["transaction_id"].isin(laundering["transaction_id"])
    ].head(MAX_EDGES - len(laundering))
    displayed = pd.concat([laundering, normal], ignore_index=True)

    displayed_accounts = set(displayed["sender_id"]) | set(displayed["recipient_id"])
    displayed_features = features[features["account_key"].astype(str).isin(displayed_accounts)].copy()
    return displayed_features, displayed


def value(row: pd.Series, column: str, default: float = 0) -> float:
    result = row.get(column, default)
    return default if pd.isna(result) else result


def create_graph(features: pd.DataFrame, transactions: pd.DataFrame) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    by_account = features.set_index("account_key")

    for account_key, row in by_account.iterrows():
        attributes = {column: value(row, column) for column in FLAG_LABELS}
        graph.add_node(
            str(account_key),
            bank_id=str(row["bank_id"]),
            account_id=str(row["account_id"]),
            typology_flag_count=int(value(row, "typology_flag_count")),
            is_laundering_account=int(value(row, "is_laundering_account")),
            unique_senders=int(value(row, "unique_senders")),
            unique_recipients=int(value(row, "unique_recipients")),
            incoming_transaction_count=int(value(row, "incoming_transaction_count")),
            outgoing_transaction_count=int(value(row, "outgoing_transaction_count")),
            structuring_event_count=int(value(row, "structuring_event_count")),
            two_account_cycle_count=int(value(row, "two_account_cycle_count")),
            three_account_cycle_count=int(value(row, "three_account_cycle_count")),
            rapid_pass_through_count=int(value(row, "rapid_pass_through_count")),
            **{key: int(flag) for key, flag in attributes.items()},
        )

    for row in transactions.itertuples(index=False):
        for node_id, bank_id, account_id in (
            (row.sender_id, row.from_bank, row.from_account),
            (row.recipient_id, row.to_bank, row.to_account),
        ):
            if not graph.has_node(str(node_id)):
                graph.add_node(
                    str(node_id), bank_id=str(bank_id), account_id=str(account_id),
                    typology_flag_count=0, is_laundering_account=0,
                    unique_senders=0, unique_recipients=0,
                    incoming_transaction_count=0, outgoing_transaction_count=0,
                    structuring_event_count=0, two_account_cycle_count=0,
                    three_account_cycle_count=0, rapid_pass_through_count=0,
                    **{key: 0 for key in FLAG_LABELS},
                )
        graph.add_edge(
            str(row.sender_id), str(row.recipient_id), key=int(row.transaction_id),
            transaction_id=int(row.transaction_id), timestamp=str(row.timestamp),
            amount_paid=float(row.amount_paid), payment_currency=str(row.payment_currency),
            amount_received=float(row.amount_received), receiving_currency=str(row.receiving_currency),
            payment_format=str(row.payment_format), is_laundering=bool(row.is_laundering),
        )
    return graph


def create_html(graph: nx.MultiDiGraph) -> None:
    positions = nx.spring_layout(graph, seed=42, iterations=100)
    view = Network(
        height="900px", width="100%", directed=True, bgcolor="#ffffff",
        font_color="#1f2937", cdn_resources="in_line",
    )

    for node_id, data in graph.nodes(data=True):
        if data["is_laundering_account"]:
            color = "#dc2626"
        elif data["typology_flag_count"]:
            color = "#f59e0b"
        else:
            color = "#94a3b8"
        node_options = {
            key: int(data[key]) for key in FLAG_LABELS
        }
        view.add_node(
            node_id, label=f"Bank {data['bank_id']}\n{data['account_id']}",
            account_id=data["account_id"], bank_id=data["bank_id"],
            color=color, size=17, physics=False,
            x=float(positions[node_id][0] * 1_600),
            y=float(positions[node_id][1] * 1_600),
            details=[
                ["Account", data["account_id"]], ["Bank", data["bank_id"]],
                ["Ground-truth laundering account", "Yes" if data["is_laundering_account"] else "No"],
                ["Structuring", "Yes" if data["structuring_flag"] else "No"],
                ["Fan-in", "Yes" if data["fan_in_flag"] else "No"],
                ["Fan-out", "Yes" if data["fan_out_flag"] else "No"],
                ["Two-way circular A-B-A", "Yes" if data["two_way_circular_flag"] else "No"],
                ["Three-way circular A-B-C-A", "Yes" if data["three_way_circular_flag"] else "No"],
                ["Rapid pass-through", "Yes" if data["rapid_pass_through_flag"] else "No"],
                ["Incoming transactions", str(data["incoming_transaction_count"])],
                ["Outgoing transactions", str(data["outgoing_transaction_count"])],
                ["Structuring events", str(data["structuring_event_count"])],
                ["Two-way cycles", str(data["two_account_cycle_count"])],
                ["Three-way cycles", str(data["three_account_cycle_count"])],
                ["Rapid pass-through events", str(data["rapid_pass_through_count"])],
            ],
            **node_options,
        )

    for sender, recipient, key, data in graph.edges(keys=True, data=True):
        view.add_edge(
            sender, recipient, id=f"transaction-{key}", transaction_id=int(key),
            color="#dc2626" if data["is_laundering"] else "#cbd5e1", arrows="to",
            details=[
                ["Transaction ID", str(key)], ["From", sender], ["To", recipient],
                ["Timestamp", data["timestamp"]],
                ["Amount paid", f"{data['amount_paid']:,.2f} {data['payment_currency']}"],
                ["Amount received", f"{data['amount_received']:,.2f} {data['receiving_currency']}"],
                ["Payment format", data["payment_format"]],
                ["Ground-truth laundering", "Yes" if data["is_laundering"] else "No"],
            ],
        )

    view.set_options('''{
      "physics": {"enabled": false},
      "edges": {"smooth": false, "width": 0.8},
      "interaction": {"hover": true, "navigationButtons": true, "multiselect": false}
    }''')

    options = '<option value="all">All accounts</option>' + "".join(
        f'<option value="{key}">{label}</option>' for key, label in FLAG_LABELS.items()
    )
    custom_html = f'''
<style>
  #controls,#graph-info{{position:fixed;z-index:10000;background:#fff;border:1px solid #d1d5db;border-radius:12px;box-shadow:0 10px 30px rgba(0,0,0,.16);font-family:Arial,sans-serif}}
  #controls{{top:20px;left:20px;width:min(430px,calc(100vw - 40px));padding:15px}}
  #graph-info{{top:20px;right:20px;width:min(370px,calc(100vw - 40px));max-height:calc(100vh - 40px);overflow-y:auto;padding:16px;display:none}}
  #controls h3,#graph-info h3{{margin:0 0 10px;font-size:16px;color:#111827}}
  .control-row{{display:grid;grid-template-columns:120px 1fr auto;gap:8px;margin-top:8px}}
  select,input,button{{border:1px solid #cbd5e1;border-radius:7px;padding:8px;font-size:13px}}
  button{{background:#2563eb;color:#fff;border-color:#2563eb;cursor:pointer}}
  #clear-button{{margin-top:8px;background:#f8fafc;color:#111827;border-color:#cbd5e1}}
  #status{{min-height:16px;margin-top:8px;color:#64748b;font-size:12px}}
  .info-row{{display:grid;grid-template-columns:145px 1fr;gap:10px;padding:7px 0;border-top:1px solid #eef0f3;font-size:13px}}
  .info-label{{color:#64748b;font-weight:bold}} .info-value{{color:#111827;overflow-wrap:anywhere}}
  .legend{{margin-top:9px;font-size:12px;color:#475569}}
</style>
<div id="controls">
  <h3>IBM AML typology graph</h3>
  <div class="control-row"><label for="typology-filter">Typology</label><select id="typology-filter">{options}</select><button id="apply-filter">Apply</button></div>
  <div class="control-row"><select id="search-type"><option value="node">Account</option><option value="edge">Transaction</option></select><input id="search-value" placeholder="Enter ID"><button id="search-button">Search</button></div>
  <button id="clear-button">Reset graph</button><div id="status"></div>
  <div class="legend">Red node/edge = ground-truth laundering · Orange node = typology indicator · Grey = no displayed indicator</div>
</div>
<div id="graph-info"><h3 id="info-title"></h3><div id="info-rows"></div></div>
<script>
const panel=document.getElementById("graph-info"),title=document.getElementById("info-title"),rows=document.getElementById("info-rows"),status=document.getElementById("status");
function showInfo(t,d){{title.textContent=t;rows.replaceChildren();for(const [k,v] of d||[]){{const r=document.createElement("div");r.className="info-row";const a=document.createElement("div");a.className="info-label";a.textContent=k;const b=document.createElement("div");b.className="info-value";b.textContent=v;r.append(a,b);rows.appendChild(r)}}panel.style.display="block"}}
function hideInfo(){{panel.style.display="none"}}
network.on("hoverNode",e=>showInfo("Bank account",nodes.get(e.node).details));network.on("blurNode",hideInfo);
network.on("hoverEdge",e=>showInfo("Transaction",edges.get(e.edge).details));network.on("blurEdge",hideInfo);
function applyFilter(){{const f=document.getElementById("typology-filter").value;const visible=new Set(),nodeUpdates=[],edgeUpdates=[];nodes.forEach(n=>{{const show=f==="all"||Number(n[f]||0)===1;if(show)visible.add(n.id);nodeUpdates.push({{id:n.id,hidden:!show}})}});edges.forEach(e=>edgeUpdates.push({{id:e.id,hidden:!(visible.has(e.from)&&visible.has(e.to))}}));nodes.update(nodeUpdates);edges.update(edgeUpdates);network.fit({{animation:false}});status.textContent=f==="all"?"Showing all displayed accounts.":"Typology filter applied."}}
function searchGraph(){{const type=document.getElementById("search-type").value,q=document.getElementById("search-value").value.trim().toLowerCase();if(!q){{status.textContent="Enter an ID.";return}}if(type==="node"){{const m=nodes.get({{filter:n=>String(n.id).toLowerCase()===q||String(n.account_id||"").toLowerCase()===q||String(n.label||"").toLowerCase().includes(q)}});if(!m.length){{status.textContent="Account not found in the displayed graph.";return}}nodes.update({{id:m[0].id,hidden:false}});network.selectNodes([m[0].id]);network.focus(m[0].id,{{scale:1.4,animation:true}});showInfo("Bank account",m[0].details);status.textContent=m.length>1?`Found ${{m.length}} accounts; showing the first.`:"Account found."}}else{{const m=edges.get({{filter:e=>String(e.transaction_id)===q||String(e.id).toLowerCase()===q}});if(!m.length){{status.textContent="Transaction not found in the displayed graph.";return}}nodes.update([{{id:m[0].from,hidden:false}},{{id:m[0].to,hidden:false}}]);edges.update({{id:m[0].id,hidden:false}});network.selectEdges([m[0].id]);network.fit({{nodes:[m[0].from,m[0].to],animation:true}});showInfo("Transaction",m[0].details);status.textContent="Transaction found."}}}}
document.getElementById("apply-filter").onclick=applyFilter;document.getElementById("search-button").onclick=searchGraph;document.getElementById("search-value").onkeydown=e=>{{if(e.key==="Enter")searchGraph()}};
document.getElementById("clear-button").onclick=()=>{{const nodeUpdates=[],edgeUpdates=[];nodes.forEach(n=>nodeUpdates.push({{id:n.id,hidden:false}}));edges.forEach(e=>edgeUpdates.push({{id:e.id,hidden:false}}));nodes.update(nodeUpdates);edges.update(edgeUpdates);network.unselectAll();network.fit({{animation:false}});document.getElementById("typology-filter").value="all";document.getElementById("search-value").value="";status.textContent="";hideInfo()}};
</script>'''
    html = view.generate_html(notebook=False).replace("</body>", custom_html + "\n</body>")
    HTML_FILE.write_text(html, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features, transactions = load_visualization_data()
    graph = create_graph(features, transactions)
    create_html(graph)
    print(f"Displayed accounts: {graph.number_of_nodes():,}")
    print(f"Displayed transactions: {graph.number_of_edges():,}")
    print(f"Visualization saved to: {HTML_FILE}")


if __name__ == "__main__":
    main()
