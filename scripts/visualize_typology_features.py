from pathlib import Path



import networkx as nx

import pandas as pd

from pyvis.network import Network





PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data" / "gvm_out"

OUTPUT_DIR = PROJECT_DIR / "outputs"



ENTITY_FILE = DATA_DIR / "ground_truth_entities.csv"

TRANSACTION_FILE = DATA_DIR / "ground_truth_transactions.csv"

FEATURE_FILE = OUTPUT_DIR / "account_typology_features.csv"



HTML_FILE = OUTPUT_DIR / "aml_typology_graph.html"





# ---------------------------------------------------------

# LOAD DATA

# ---------------------------------------------------------



entities = pd.read_csv(

    ENTITY_FILE,

    dtype={

        "iban": "string",

        "bic": "string",

        "blz": "string",

    },

)



transactions = pd.read_csv(

    TRANSACTION_FILE

)



features = pd.read_csv(

    FEATURE_FILE

)





# Combine account information and typology features.

accounts = entities.merge(

    features,

    on="account_id",

    how="left",

)



accounts = accounts.fillna(0)



def optional_text(value):

    text = str(value).strip()

    if text in {"", "0", "0.0", "nan", "NaT"}:

        return "Not available"

    return text


def format_berlin_timestamp(value):

    """Convert a UTC/ISO transaction timestamp to Berlin local time."""

    try:

        timestamp = pd.to_datetime(value, utc=True)

        berlin_time = timestamp.tz_convert("Europe/Berlin")

        return berlin_time.strftime(

            "%Y-%m-%d %H:%M:%S %Z"

        )

    except (TypeError, ValueError):

        return str(value)





# ---------------------------------------------------------

# BUILD GRAPH

# ---------------------------------------------------------



graph = nx.MultiDiGraph()



for row in accounts.itertuples(index=False):

    graph.add_node(

        int(row.account_id),

        institution_name=str(row.institution_name),

        account_type=str(row.account_type),

        iban=str(row.iban),

        balance=float(row.balance),

        structuring_flag=int(row.structuring_flag),

        fan_in_flag=int(row.fan_in_flag),

        fan_out_flag=int(row.fan_out_flag),

        circular_transfer_flag=int(

            row.circular_transfer_flag

        ),

        two_way_circular_flag=int(

            row.two_way_circular_flag

        ),

        three_way_circular_flag=int(

            row.three_way_circular_flag

        ),

        rapid_pass_through_flag=int(

            row.rapid_pass_through_flag

        ),

        typology_flag_count=int(

            row.typology_flag_count

        ),

        unique_senders=int(row.unique_senders),

        unique_recipients=int(row.unique_recipients),

        structuring_window_count=int(

            row.structuring_window_count

        ),

        circular_transfer_count=int(

            row.circular_transfer_count

        ),

        two_account_cycle_count=int(

            row.two_account_cycle_count

        ),

        three_account_cycle_count=int(

            row.three_account_cycle_count

        ),

        rapid_pass_through_count=int(

            row.rapid_pass_through_count

        ),

        rapid_incoming_timestamp=optional_text(

            row.rapid_incoming_timestamp

        ),

        rapid_outgoing_timestamp=optional_text(

            row.rapid_outgoing_timestamp

        ),

        rapid_example_delay_hours=float(

            row.rapid_example_delay_hours

        ),

        average_rapid_pass_through_hours=float(

            row.average_rapid_pass_through_hours

        ),

    )





for row in transactions.itertuples(index=False):

    graph.add_edge(

        int(row.orig_acct),

        int(row.bene_acct),

        key=int(row.tran_id),

        tran_id=int(row.tran_id),

        amount=float(row.amount),

        timestamp=str(row.timestamp),

        tx_type=str(row.tx_type),

        is_sar=bool(row.is_sar),

    )





# Stable positions.

positions = nx.spring_layout(

    graph,

    seed=42,

    iterations=150,

)





# ---------------------------------------------------------

# CREATE PYVIS GRAPH

# ---------------------------------------------------------



view = Network(

    height="850px",

    width="100%",

    directed=True,

    bgcolor="#ffffff",

    font_color="#1f2937",

    cdn_resources="in_line",

)





def account_color(flag_count):

    if flag_count >= 3:

        return "#991b1b"



    if flag_count == 2:

        return "#dc2626"



    if flag_count == 1:

        return "#f97316"



    return "#cbd5e1"





for account_id, data in graph.nodes(data=True):

    flag_count = data["typology_flag_count"]



    view.add_node(

        account_id,

        label=f"Account {account_id}",

        color=account_color(flag_count),

        size=16 + (flag_count * 5),

        x=float(positions[account_id][0] * 1800),

        y=float(positions[account_id][1] * 1800),

        physics=False,



        structuring_flag=data["structuring_flag"],

        fan_in_flag=data["fan_in_flag"],

        fan_out_flag=data["fan_out_flag"],

        circular_transfer_flag=(

            data["circular_transfer_flag"]

        ),

        two_way_circular_flag=(

            data["two_way_circular_flag"]

        ),

        three_way_circular_flag=(

            data["three_way_circular_flag"]

        ),

        rapid_pass_through_flag=(

            data["rapid_pass_through_flag"]

        ),



        details=[

            ["Account ID", str(account_id)],

            ["Bank", data["institution_name"]],

            ["Account type", data["account_type"]],

            ["IBAN", data["iban"]],

            ["Balance", f"€{data['balance']:,.2f}"],

            [

                "Structuring",

                (

                    "Yes"

                    if data["structuring_flag"]

                    else "No"

                ),

            ],

            [

                "Structuring windows",

                str(data["structuring_window_count"]),

            ],

            [

                "Fan-in",

                (

                    "Yes"

                    if data["fan_in_flag"]

                    else "No"

                ),

            ],

            [

                "Fan-out",

                (

                    "Yes"

                    if data["fan_out_flag"]

                    else "No"

                ),

            ],

            [

                "Circular transfers",

                (

                    "Yes"

                    if data["circular_transfer_flag"]

                    else "No"

                ),

            ],

            [

                "Two-way A-B-A",

                (

                    "Yes"

                    if data["two_way_circular_flag"]

                    else "No"

                ),

            ],

            [

                "Two-way cycle count",

                str(data["two_account_cycle_count"]),

            ],

            [

                "Three-way A-B-C-A",

                (

                    "Yes"

                    if data["three_way_circular_flag"]

                    else "No"

                ),

            ],

            [

                "Three-way cycle count",

                str(data["three_account_cycle_count"]),

            ],

            [

                "Cycle count",

                str(data["circular_transfer_count"]),

            ],

            [

                "Rapid pass-through",

                (

                    "Yes"

                    if data["rapid_pass_through_flag"]

                    else "No"

                ),

            ],

            [

                "Rapid pass-through count",

                str(data["rapid_pass_through_count"]),

            ],

        ],

    )





for sender, recipient, tran_id, data in graph.edges(

    keys=True,

    data=True,

):

    view.add_edge(

        sender,

        recipient,

        id=f"transaction-{tran_id}",

        transaction_id=int(tran_id),

        color=(

            "#dc2626"

            if data["is_sar"]

            else "#cbd5e1"

        ),

        arrows="to",

        details=[

            ["Transaction ID", str(tran_id)],

            ["From account", str(sender)],

            ["To account", str(recipient)],

            ["Amount", f"€{data['amount']:,.2f}"],

            [

                "Timestamp (Berlin)",

                format_berlin_timestamp(data["timestamp"]),

            ],

            ["Type", data["tx_type"]],

            [

                "Ground-truth suspicious",

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

        "width": 0.7

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





# ---------------------------------------------------------

# TYPOLOGY CONTROL AND INFORMATION PANEL

# ---------------------------------------------------------



CUSTOM_HTML = """

<style>

  #typology-controls {

    position: fixed;

    top: 20px;

    left: 20px;

    z-index: 10000;

    width: 260px;

    padding: 15px;

    background: white;

    border: 1px solid #d1d5db;

    border-radius: 12px;

    box-shadow: 0 10px 30px rgba(0,0,0,0.18);

    font-family: Arial, sans-serif;

  }



  #typology-controls h3 {

    margin: 0 0 12px;

    font-size: 17px;

  }



  #typology-select {

    width: 100%;

    padding: 8px;

    box-sizing: border-box;

    border: 1px solid #cbd5e1;

    border-radius: 6px;

  }



  #typology-controls button {

    width: 100%;

    padding: 8px;

    margin-top: 8px;

    cursor: pointer;

    border: 1px solid #cbd5e1;

    border-radius: 6px;

    background: #f8fafc;

  }



  #typology-summary {

    margin-top: 10px;

    padding: 8px;

    background: #f1f5f9;

    border-radius: 6px;

    color: #475569;

    font-size: 12px;

  }



  #typology-legend {

    margin-top: 10px;

    font-size: 12px;

    line-height: 1.7;

  }



  .legend-circle {

    display: inline-block;

    width: 11px;

    height: 11px;

    margin-right: 5px;

    border-radius: 50%;

  }



  #graph-info {

    position: fixed;

    top: 20px;

    right: 20px;

    z-index: 10000;

    width: 340px;

    max-height: calc(100vh - 40px);

    overflow-y: auto;

    padding: 16px;

    display: none;

    background: white;

    border: 1px solid #d1d5db;

    border-radius: 12px;

    box-shadow: 0 10px 30px rgba(0,0,0,0.18);

    font-family: Arial, sans-serif;

  }



  #graph-info h3 {

    margin: 0 0 12px;

  }



  .info-row {

    display: grid;

    grid-template-columns: 145px 1fr;

    gap: 8px;

    padding: 6px 0;

    border-top: 1px solid #eef0f3;

    font-size: 13px;

  }



  .info-label {

    color: #64748b;

    font-weight: bold;

  }



  .info-value {

    overflow-wrap: anywhere;

  }

</style>



<div id="typology-controls">

  <h3>AML typology view</h3>



  <select id="typology-select">

    <option value="all">

      All accounts

    </option>



    <option value="any">

      All typology indicators

    </option>



    <option value="structuring_flag">

      Structuring

    </option>



    <option value="fan_in_flag">

      Fan-in

    </option>



    <option value="fan_out_flag">

      Fan-out

    </option>



    <option value="circular_transfer_flag">

      Circular transfers

    </option>



    <option value="two_way_circular_flag">

      Two-way circular (A-B-A)

    </option>



    <option value="three_way_circular_flag">

      Three-way circular (A-B-C-A)

    </option>



    <option value="rapid_pass_through_flag">

      Rapid pass-through

    </option>

  </select>



  <button id="apply-typology">

    Apply view

  </button>



  <button id="show-full-graph">

    Show full graph

  </button>



  <div id="typology-summary">

    Full graph displayed.

  </div>



  <div id="typology-legend">

    <div>

      <span

        class="legend-circle"

        style="background:#cbd5e1"

      ></span>

      No indicator

    </div>



    <div>

      <span

        class="legend-circle"

        style="background:#f97316"

      ></span>

      One indicator

    </div>



    <div>

      <span

        class="legend-circle"

        style="background:#dc2626"

      ></span>

      Two indicators

    </div>



    <div>

      <span

        class="legend-circle"

        style="background:#991b1b"

      ></span>

      Three or more indicators

    </div>

  </div>

</div>



<div id="graph-info">

  <h3 id="graph-info-title"></h3>

  <div id="graph-info-rows"></div>

</div>



<script>

  const completeNodes = nodes

    .get()

    .map(node => ({...node}));



  const completeEdges = edges

    .get()

    .map(edge => ({...edge}));



  const typologySelect =

    document.getElementById("typology-select");



  const typologySummary =

    document.getElementById("typology-summary");



  const infoPanel =

    document.getElementById("graph-info");



  const infoTitle =

    document.getElementById("graph-info-title");



  const infoRows =

    document.getElementById("graph-info-rows");





  function showDetails(title, details) {

    infoTitle.textContent = title;

    infoRows.replaceChildren();



    for (const [label, value] of details || []) {

      const row = document.createElement("div");

      row.className = "info-row";



      const labelElement =

        document.createElement("div");



      labelElement.className = "info-label";

      labelElement.textContent = label;



      const valueElement =

        document.createElement("div");



      valueElement.className = "info-value";

      valueElement.textContent = value;



      row.append(labelElement, valueElement);

      infoRows.appendChild(row);

    }



    infoPanel.style.display = "block";

  }





  function displayGraph(

    visibleNodes,

    visibleEdges

  ) {

    network.unselectAll();

    edges.clear();

    nodes.clear();



    nodes.add(

      visibleNodes.map(node => ({...node}))

    );



    edges.add(

      visibleEdges.map(edge => ({...edge}))

    );



    infoPanel.style.display = "none";



    setTimeout(function () {

      network.fit({

        animation: false

      });

    }, 50);

  }





  document

    .getElementById("apply-typology")

    .addEventListener("click", function () {

      const selectedFeature =

        typologySelect.value;



      if (selectedFeature === "all") {

        displayGraph(

          completeNodes,

          completeEdges

        );



        typologySummary.textContent =

          "Full graph displayed.";



        return;

      }



      let flaggedNodes;



      if (selectedFeature === "any") {

        flaggedNodes = completeNodes.filter(

          node =>

            Number(node.structuring_flag) === 1 ||

            Number(node.fan_in_flag) === 1 ||

            Number(node.fan_out_flag) === 1 ||

            Number(node.circular_transfer_flag) === 1 ||

            Number(node.rapid_pass_through_flag) === 1

        );

      } else {

        flaggedNodes = completeNodes.filter(

          node =>

            Number(node[selectedFeature]) === 1

        );

      }



      const flaggedIds = new Set(

        flaggedNodes.map(

          node => String(node.id)

        )

      );



      /*

       * Keep transactions connected to flagged accounts.

       */

      const connectedEdges = completeEdges.filter(

        edge =>

          flaggedIds.has(String(edge.from)) ||

          flaggedIds.has(String(edge.to))

      );



      /*

       * Include counterparties so that the displayed

       * transactions still have both endpoints.

       */

      const visibleIds = new Set(flaggedIds);



      connectedEdges.forEach(edge => {

        visibleIds.add(String(edge.from));

        visibleIds.add(String(edge.to));

      });



      const visibleNodes = completeNodes.filter(

        node =>

          visibleIds.has(String(node.id))

      );



      displayGraph(

        visibleNodes,

        connectedEdges

      );



      typologySummary.textContent =

        `${flaggedNodes.length} accounts have ` +

        `the selected indicator. ` +

        `${connectedEdges.length} connected ` +

        `transactions are displayed.`;

    });





  document

    .getElementById("show-full-graph")

    .addEventListener("click", function () {

      displayGraph(

        completeNodes,

        completeEdges

      );



      typologySelect.value = "all";



      typologySummary.textContent =

        "Full graph displayed.";

    });





  network.on("hoverNode", function (event) {

    const node = nodes.get(event.node);



    if (node) {

      showDetails(

        "Account typology features",

        node.details

      );

    }

  });





  network.on("blurNode", function () {

    infoPanel.style.display = "none";

  });





  network.on("hoverEdge", function (event) {

    const edge = edges.get(event.edge);



    if (edge) {

      showDetails(

        "Transaction",

        edge.details

      );

    }

  });





  network.on("blurEdge", function () {

    infoPanel.style.display = "none";

  });

</script>

"""





# ---------------------------------------------------------

# SAVE HTML

# ---------------------------------------------------------



OUTPUT_DIR.mkdir(

    parents=True,

    exist_ok=True,

)



html = view.generate_html(

    notebook=False

)



html = html.replace(

    "</body>",

    CUSTOM_HTML + "\n</body>",

)



HTML_FILE.write_text(

    html,

    encoding="utf-8",

)



print(

    f"AML typology visualization saved to: "

    f"{HTML_FILE}"

)
