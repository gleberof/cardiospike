import json
from pathlib import Path
from typing import List

import pandas as pd
import plotly
import requests
from flask import Flask, render_template, request

from cardiospike.api import API_PORT
from cardiospike.utils.visualization import plot_rr
from cardiospike.web import API_HOST, STATIC_DIR

from cardiospike import TEST_PATH, WELLTORY_PATH


df = pd.read_csv(Path(TEST_PATH))
wt = pd.read_csv(Path(WELLTORY_PATH))

df = pd.concat((df,wt))

users = [str(u) for u in df.id.unique()]


app = Flask(__name__, static_folder=STATIC_DIR)

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


def gm(sample="8"):
    t = df.loc[df["id"] == int(sample)].sort_values("time").reset_index(drop=True)
    results = get_predictions(sample, t["x"].tolist())

    anomaly_thresh = results["anomaly_thresh"]

    t["anomaly_proba"] = results["anomaly_proba"]
    t["error"] = results["errors"]

    fig = plot_rr(t, anomaly_thresh=anomaly_thresh)

    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graph_json


if __name__ == "__main__":
    app.run()
