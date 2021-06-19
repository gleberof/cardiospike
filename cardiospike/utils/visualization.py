import plotly.graph_objs as go


def plot_rr(t, anomaly_thresh=0.4):
    fig = go.Figure()
    fig.add_scatter(x=t["time"], y=t["x"], line_color="#e83e8c", line_width=3, name="RR Interval")

    mask = t["anomaly_proba"] > anomaly_thresh

    fig.add_scatter(
        x=t["time"][mask],
        y=t["x"][mask],
        mode="markers",
        marker_color=t["anomaly_proba"][mask],
        marker_size=15 * t["anomaly_proba"][mask],
        marker_colorscale=[[0.0, "#e83e8c"], [1.0, "#ffc107"]],
        marker_line=dict(color="#dc3545", width=3 * t["anomaly_proba"][mask]),
        marker_opacity=t["anomaly_proba"][mask],
        name="Anomaly",
    )
    fig.update_layout(
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="#f8f9fa",
        xaxis_gridcolor="#007bff",
        yaxis_gridcolor="#007bff",
        xaxis_color="#007bff",
        yaxis_color="#007bff",
        xaxis_title_text_color="#007bff",
        xaxis_title_text="Время, мс",
    )

    return fig
