import plotly.graph_objs as go


def anomaly_count(array_predictions):
    # count -> 3
    count = 0
    i = 0
    while i < len(array_predictions) - 1:
        if array_predictions[i]:
            while array_predictions[i] and array_predictions[i + 1]:
                i += 1
            count += 1
        i += 1
    return count


def plot_rr(t, anomaly_thresh=0.4):
    fig = go.Figure()
    fig.add_scatter(x=t["time"], y=t["x"], line_color="#e83e8c", line_width=3, name="RR-интервалы")

    mask = t["anomaly_proba"] > anomaly_thresh

    num_anomalies = anomaly_count(mask)

    factor = ((t["anomaly_proba"] - anomaly_thresh) / max(t["anomaly_proba"] - anomaly_thresh))[mask]

    plot_name = (
        "RR Ритмограмма. Все чисто!"
        if num_anomalies == 0
        else f"RR Ритмограмма. ❗ Обнаружено {num_anomalies} аномалий ❗"
    )

    fig.add_scatter(
        x=t["time"][mask],
        y=t["x"][mask],
        mode="markers",
        marker_color=factor,
        marker_size=15 * factor,
        marker_colorscale=[[0.0, "#e83e8c"], [1.0, "#ffc107"]],
        marker_line=dict(color="#dc3545", width=3 * factor),
        marker_opacity=factor,
        name="Аномалии",
        hovertext=[f"Вероятность аномалии: {p:.0%}" for p in t["anomaly_proba"][mask]],
    )
    fig.update_layout(
        title_text=plot_name,
        title_font_color="#1940ff",
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
        gridcolor="#1940ff",
        color="#1940ff",
        linewidth=3,
        title_font_color="#1940ff",
        title_font_size=15,
        zerolinecolor="#1940ff",
        zerolinewidth=5,
        title_font_family="Segoe UI",
    )

    fig.update_xaxes(**axis_config)
    fig.update_yaxes(**axis_config)

    return fig, num_anomalies
