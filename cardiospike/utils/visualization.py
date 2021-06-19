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
        title_text="RR Ритмограмма",
        title_font_color="#007bff",
        title_font_size=20,
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="#f8f9fa",
        xaxis_title_text="Время, мс",
        yaxis_title_text="RR-интервал, мс",
        margin=go.layout.Margin(
            l=100,  # left margin
            r=0,  # right margin
            b=0,  # bottom margin
            t=40,  # top margin
        ),
    )

    axis_config = dict(
        gridcolor="#007bff",
        color="#007bff",
        linewidth=3,
        title_font_color="#007bff",
        title_font_size=15,
        zerolinecolor="#007bff",
        zerolinewidth=5,
        title_font_family="Segoe UI",
    )

    fig.update_xaxes(**axis_config)
    fig.update_yaxes(**axis_config)

    return fig