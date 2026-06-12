"""
clustering.py
─────────────
KMeans clustering helpers: elbow-method analysis and cluster visualisation.
Returns Plotly figures so the caller controls rendering.

Render in Streamlit with:
    st.plotly_chart(fig, use_container_width=True)
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans


# ──────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _clean_pair(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """Return a clean two-column DataFrame with no NaNs."""
    return df[[x_col, y_col]].dropna().reset_index(drop=True)


def _apply_theme(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply a consistent, clean theme to every figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=13),
        margin=dict(l=50, r=30, t=50, b=50),
        hoverlabel=dict(
            bgcolor="white",
            font_size=13,
            font_family="Inter, sans-serif",
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
#  Elbow chart
# ──────────────────────────────────────────────────────────────────────────────

def compute_inertias(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    max_k: int = 10,
) -> dict[int, float]:
    """Return {k: inertia} for k in 1..max_k."""
    data = _clean_pair(df, x_col, y_col)
    return {
        k: KMeans(n_clusters=k, n_init=10, random_state=42).fit(data).inertia_
        for k in range(1, max_k + 1)
    }


def plot_elbow(inertias: dict[int, float]) -> go.Figure:
    """
    Interactive elbow chart. Hover shows the exact inertia for each k.
    The user can visually identify where the curve bends (the elbow).
    """
    ks = list(inertias.keys())
    vals = list(inertias.values())

    fig = go.Figure()

    # Shaded area under the curve for visual weight
    fig.add_trace(
        go.Scatter(
            x=ks,
            y=vals,
            fill="tozeroy",
            fillcolor="rgba(52, 152, 219, 0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Main line + markers
    fig.add_trace(
        go.Scatter(
            x=ks,
            y=vals,
            mode="lines+markers",
            line=dict(color="#3498db", width=2.5),
            marker=dict(
                size=9,
                color="#3498db",
                line=dict(color="white", width=2),
            ),
            name="Inertia",
            hovertemplate="<b>k = %{x}</b><br>Inertia: %{y:,.1f}<extra></extra>",
        )
    )

    fig.update_xaxes(title_text="Number of Clusters (k)", tickvals=ks)
    fig.update_yaxes(title_text="Inertia")
    _apply_theme(fig, "Elbow Method — Optimal k")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
#  Cluster scatter plot
# ──────────────────────────────────────────────────────────────────────────────

def fit_and_plot_clusters(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    n_clusters: int,
) -> go.Figure:
    """
    KMeans scatter plot with cluster colouring.
    Hover shows the x/y values, the cluster label, and the row index.
    Centroids are plotted as larger star markers.
    """
    data = _clean_pair(df, x_col, y_col).copy()
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = model.fit_predict(data)
    centroids = model.cluster_centers_

    data["Cluster"] = labels.astype(str)
    data["Row"] = data.index

    fig = px.scatter(
        data,
        x=x_col,
        y=y_col,
        color="Cluster",
        color_discrete_sequence=px.colors.qualitative.T10,
        opacity=0.75,
        hover_data={"Row": True, "Cluster": True},
        title=f"KMeans (k={n_clusters}) — {x_col} vs {y_col}",
    )
    fig.update_traces(
        marker=dict(size=8),
        hovertemplate=(
            f"<b>{x_col}:</b> %{{x}}<br>"
            f"<b>{y_col}:</b> %{{y}}<br>"
            "<b>Cluster:</b> %{customdata[1]}<br>"
            "<b>Row:</b> %{customdata[0]}<extra></extra>"
        ),
    )

    # Centroid markers
    for i, (cx, cy) in enumerate(centroids):
        fig.add_trace(
            go.Scatter(
                x=[cx],
                y=[cy],
                mode="markers",
                marker=dict(
                    symbol="star",
                    size=18,
                    color=px.colors.qualitative.T10[i % len(px.colors.qualitative.T10)],
                    line=dict(color="white", width=1.5),
                ),
                name=f"Centroid {i}",
                hovertemplate=(
                    f"<b>Centroid {i}</b><br>"
                    f"{x_col}: {cx:.2f}<br>"
                    f"{y_col}: {cy:.2f}<extra></extra>"
                ),
            )
        )

    _apply_theme(fig, f"KMeans (k={n_clusters}) — {x_col} vs {y_col}")
    fig.update_layout(legend_title_text="Cluster")
    return fig