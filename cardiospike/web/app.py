import json
import os
from pathlib import Path
from typing import List

import pandas as pd
import plotly
import plotly.graph_objects as go
import requests
from flask import Flask, render_template, request

from cardiospike.api import API_PORT
from cardiospike.web import API_HOST

file_path = Path(os.path.realpath(__file__)).parent.parent.parent.absolute()

app = Flask(__name__)

df = pd.read_csv(Path(f"{file_path}/data/train.csv"))
users = [str(u) for u in df.id.unique()]


API_ENDPOINT = f"http://{API_HOST}:{API_PORT}"
PREDICT_ENDPOINT = f"{API_ENDPOINT}/predict"


@app.route("/callback", methods=["POST", "GET"])
def cb():
    return gm(request.args.get("data"))


@app.route("/")
def index():
    return render_template("index.html", graphJSON=gm(), users=users)


def get_predictions(study: str, sequence: List[int]):
    json_data = json.dumps(
        {
            "study": study,
            "sequence": sequence,
        }
    )
    headers = {"Content-Type": "application/json"}
    session = requests.Session()
    session.trust_env = False
    response = session.post(url=PREDICT_ENDPOINT, headers=headers, data=json_data)
    if response.status_code == 200:
        return response.json()


def gm(sample="1"):
    t = df.loc[df["id"] == int(sample)].sort_values("time").reset_index(drop=True)
    fig = go.Figure()
    fig.update_layout(title=f"Sample {sample}")

    results = get_predictions(sample, t["x"].tolist())

    fig.add_trace(go.Scatter(x=t["time"], y=t["x"], mode="lines", name="R-R"))
    fig.update_traces(line=dict(color="blue", width=0.3))
    t["predictions"] = results["predictions"]
    t["errors"] = results["errors"]

    qt = t.loc[t.y == 1].reset_index(drop=True)
    fig.add_trace(
        go.Scatter(
            x=qt["time"],
            y=qt["x"],
            mode="markers",
            name="spikes",
            opacity=0.8,
            marker=dict(
                size=10,
            ),
        )
    )

    pred = t.loc[t["predictions"] > 0.99].reset_index(drop=True)
    fig.add_trace(go.Scatter(x=pred["time"], y=pred["x"], mode="markers", name="predicts"))

    errs = t.loc[t["errors"] > 0.99].reset_index(drop=True)
    fig.add_trace(go.Scatter(x=errs["time"], y=errs["x"], mode="markers", name="errors", marker_symbol="x"))

    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graph_json


if __name__ == "__main__":
    app.run()
