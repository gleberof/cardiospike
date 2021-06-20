import numpy as np
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

    is_anomaly_mask = t["anomaly_proba"] > anomaly_thresh
    is_error_mask = t["error"] == 1

    num_anomalies = anomaly_count(is_anomaly_mask)

    anomaly_proba = t["anomaly_proba"].values
    anomaly_color = np.sqrt(anomaly_proba)
    anomaly_size = anomaly_proba.copy() * 20
    anomaly_size[~is_anomaly_mask] = 0

    anomaly_opacity = is_anomaly_mask.astype("int")

    plot_name = "RR Ритмограмма. Все чисто!" if num_anomalies == 0 else f"RR Ритмограмма. ❗ Аномалий: {num_anomalies} ❗"
    anomaly_text = [f"{p:.0%}" for p in anomaly_proba]

    fig = go.Figure()

    fig.add_scatter(x=t["time"], y=t["x"], line_color="#e83e8c", line_width=3, name="RR-интервал")
    fig.add_scatter(
        x=t["time"],
        y=t["x"],
        mode="markers",
        marker_color=anomaly_color,
        marker_size=anomaly_size,
        marker_colorscale=[[0.0, "#e83e8c"], [anomaly_thresh, "#e83e8c"], [1.0, "#ffc107"]],
        marker_line=dict(color="#dc3545", width=3 * anomaly_proba),
        marker_opacity=anomaly_opacity,
        name="Аномалия",
        text=anomaly_text,
        hovertemplate="%{text}",
    )
    fig.add_scatter(
        x=t["time"][is_error_mask],
        y=t["x"][is_error_mask],
        mode="markers",
        marker_color="#20c997",
        marker_size=13,
        marker_line_width=2,
        marker_symbol="x",
        name="Ошибка измерения",
        hoverinfo="skip",
    )
    fig.update_layout(
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.09,
            bgcolor="#f8f9fa",
            bordercolor="#1940ff",
            borderwidth=3,
            title_text="Легенда",
            title_font_color="#1940ff",
            itemsizing="constant",
            font_color="#1940ff",
        ),
        title={"text": plot_name, "font_size": 20},
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
        hovermode="x unified",
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
