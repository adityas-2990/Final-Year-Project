"""
visualizations.py
─────────────────
All Plotly figure-building functions.
Every function returns a ``plotly.graph_objects.Figure`` so callers
(Streamlit or tests) decide how to render / save it.

Render in Streamlit with:
    st.plotly_chart(fig, use_container_width=True)
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

from data_utils import convert_categorical_to_numeric


# ──────────────────────────────────────────────────────────────────────────────
#  Shared theme helper
# ──────────────────────────────────────────────────────────────────────────────

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
#  Missing-values heatmap
# ──────────────────────────────────────────────────────────────────────────────

def plot_missing_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Interactive heatmap of missing values.
    Hover shows exact (row, column) and whether the cell is missing.
    """
    missing_matrix = df.isnull().astype(int)

    fig = px.imshow(
        missing_matrix,
        color_continuous_scale=["#e8f4f8", "#e74c3c"],
        aspect="auto",
        labels=dict(color="Missing"),
        title="Missing Values Heatmap",
    )

    fig.update_coloraxes(showscale=False)

    fig.update_xaxes(tickangle=-45)

    fig.update_traces(
        hovertemplate=(
            "<b>Row %{y}</b><br>"
            "Column: %{x}<br>"
            "Missing: %{z}<extra></extra>"
        )
    )

    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial",
            font_color="black"
        )
    )

    _apply_theme(fig, "Missing Values Heatmap")

    return fig

# ──────────────────────────────────────────────────────────────────────────────
#  Outlier box-plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_boxplot(df: pd.DataFrame, column: str) -> go.Figure:
    """
    Interactive box plot with cleaner hover labels and better spacing.
    """

    fig = px.box(
        df,
        x=column,
        points="outliers",   # Show only outliers instead of all points
        color_discrete_sequence=["#3498db"],
        title=f"Boxplot — {column}",
    )

    fig.update_traces(
        marker=dict(size=6, opacity=0.7),

        hovertemplate=(
            f"<b>{column}</b><br>"
            "Value: %{x}<extra></extra>"
        ),

        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_color="black"
        )
    )

    fig.update_layout(
        margin=dict(t=80, b=60, l=40, r=40),
        xaxis_title=column,
        yaxis_title="",
    )

    _apply_theme(fig, f"Boxplot — {column}")

    return fig


# ──────────────────────────────────────────────────────────────────────────────
#  Bar chart (categorical × numeric)
# ──────────────────────────────────────────────────────────────────────────────

def plot_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> go.Figure:
    """
    Interactive grouped bar chart with improved readability.
    """

    grouped = (
        df.groupby(y_col)[x_col]
        .mean()
        .reset_index()
        .sort_values(x_col, ascending=False)
    )

    fig = px.bar(
        grouped,
        x=y_col,
        y=x_col,
        color=x_col,
        color_continuous_scale="Blues",
        title=f"{x_col} by {y_col}",
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{x}</b><br>"
            f"{x_col}: %{{y:.2f}}"
            "<extra></extra>"
        ),

        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_color="black"
        )
    )

    fig.update_layout(
        xaxis_title=y_col,
        yaxis_title=x_col,

        height=600,

        margin=dict(
            t=80,
            b=150,
            l=60,
            r=40
        ),

        xaxis=dict(
            tickangle=-35,
            automargin=True
        )
    )

    fig.update_coloraxes(showscale=False)

    _apply_theme(fig, f"{x_col} by {y_col}")

    return fig


# ──────────────────────────────────────────────────────────────────────────────
#  Numeric scatter / line / bar
# ──────────────────────────────────────────────────────────────────────────────

def plot_numeric_graph(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    graph_type: str,  # "Scatter Plot" | "Line Plot" | "Bar Plot"
) -> go.Figure:
    """
    Interactive scatter, line, or bar chart between two numeric columns.
    Scatter includes a trendline; hover shows both axis values.
    """
    if graph_type == "Scatter Plot":
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            trendline="ols",
            trendline_color_override="#e74c3c",
            color_discrete_sequence=["#3498db"],
            opacity=0.7,
            title=f"Scatter: {y_col} vs {x_col}",
        )
        fig.update_traces(
            hovertemplate=f"<b>{x_col}:</b> %{{x}}<br><b>{y_col}:</b> %{{y}}<extra></extra>",
            selector=dict(mode="markers"),
        )

    elif graph_type == "Line Plot":
        df_sorted = df[[x_col, y_col]].dropna().sort_values(x_col)
        fig = px.line(
            df_sorted,
            x=x_col,
            y=y_col,
            color_discrete_sequence=["#3498db"],
            title=f"Line: {y_col} vs {x_col}",
        )
        fig.update_traces(
            hovertemplate=f"<b>{x_col}:</b> %{{x}}<br><b>{y_col}:</b> %{{y}}<extra></extra>"
        )

    else:  # Bar Plot
        fig = px.bar(
            df,
            x=x_col,
            y=y_col,
            color_discrete_sequence=["#3498db"],
            title=f"Bar: {y_col} vs {x_col}",
        )
        fig.update_traces(
            hovertemplate=f"<b>{x_col}:</b> %{{x}}<br><b>{y_col}:</b> %{{y}}<extra></extra>"
        )

    _apply_theme(fig, fig.layout.title.text)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
#  Correlation heatmap
# ──────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure | None:
    """
    Interactive annotated correlation heatmap.
    Hover shows exact correlation value and column names.
    Returns None if fewer than 2 numeric columns exist.
    """

    num_df = df.select_dtypes(include=["int64", "float64"])

    if num_df.shape[1] < 2:
        return None

    corr = num_df.corr().round(2)

    fig = px.imshow(
        corr,

        text_auto=True,

        color_continuous_scale="RdBu_r",

        zmin=-1,
        zmax=1,

        aspect="auto",

        title="Correlation Heatmap",
    )

    fig.update_traces(
        hovertemplate=(
            "<b>%{x}</b> × <b>%{y}</b><br>"
            "Correlation: %{z:.2f}"
            "<extra></extra>"
        ),

        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_color="black"
        )
    )

    fig.update_xaxes(
        tickangle=-45,
        automargin=True
    )

    fig.update_yaxes(
        automargin=True
    )

    fig.update_layout(
        height=700,

        margin=dict(
            t=80,
            b=120,
            l=120,
            r=40
        ),

        coloraxis_colorbar=dict(
            title="Correlation"
        )
    )

    _apply_theme(fig, "Correlation Heatmap")

    return fig


# ──────────────────────────────────────────────────────────────────────────────
#  Pie / Donut chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_pie_chart(
    df: pd.DataFrame,
    column: str,
    style: str = "Pie Chart",   # "Pie Chart" | "Donut Chart"
    show_pct: bool = True,
) -> go.Figure:
    """
    Interactive pie or donut chart. Hover shows count, percentage,
    and category label. Clicking a slice isolates it.
    """
    counts = df[column].dropna().astype(str).value_counts().reset_index()
    counts.columns = [column, "count"]

    hole = 0.45 if style == "Donut Chart" else 0.0

    fig = px.pie(
        counts,
        names=column,
        values="count",
        hole=hole,
        title=f"{style} — {column}",
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    fig.update_traces(
        textinfo="percent+label" if show_pct else "label",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
        pull=[0.03] * len(counts),
    )

    if style == "Donut Chart":
        fig.add_annotation(
            text=f"<b>Total<br>{counts['count'].sum():,}</b>",
            x=0.5, y=0.5,
            font=dict(size=14),
            showarrow=False,
        )

    _apply_theme(fig, f"{style} — {column}")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
#  Histogram + KDE
# ──────────────────────────────────────────────────────────────────────────────

def plot_histogram(
    df: pd.DataFrame,
    column: str,
    bins: int = 30,
    bar_color: str = "#4C72B0",
    kde_color: str = "#DD4444",
    show_kde: bool = True,
    show_mean_median: bool = True,
) -> go.Figure:
    """
    Histogram with optional KDE overlay and mean/median lines.
    Hover shows bin range and count.
    """
    col_data = df[column].dropna()

    fig = go.Figure()

    # Histogram bars
    fig.add_trace(
        go.Histogram(
            x=col_data,
            nbinsx=bins,
            marker_color=bar_color,
            opacity=0.75,
            name="Count",
            hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
        )
    )

    if show_kde:
        # Normalise histogram to density so KDE overlays correctly
        fig.update_traces(histnorm="probability density", selector=dict(type="histogram"))

        kde = gaussian_kde(col_data)
        x_range = np.linspace(col_data.min(), col_data.max(), 300)
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=kde(x_range),
                mode="lines",
                line=dict(color=kde_color, width=2.5),
                name="KDE",
                hovertemplate="x: %{x:.2f}<br>Density: %{y:.4f}<extra></extra>",
            )
        )

    if show_mean_median:
        mean_val = col_data.mean()
        median_val = col_data.median()

        fig.add_vline(
            x=mean_val,
            line=dict(color="#2ecc71", width=2, dash="dash"),
            annotation_text=f"Mean: {mean_val:.2f}",
            annotation_position="top right",
        )
        fig.add_vline(
            x=median_val,
            line=dict(color="#f39c12", width=2, dash="dashdot"),
            annotation_text=f"Median: {median_val:.2f}",
            annotation_position="top left",
        )

    _apply_theme(fig, f"Distribution of '{column}'")
    fig.update_layout(bargap=0.05, showlegend=show_kde)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
#  Regression plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_regression(df: pd.DataFrame, x_col: str, y_col: str) -> go.Figure:
    """
    Scatter plot with OLS regression line and 95 % confidence band.
    Hover shows both axis values for each point.
    """
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        trendline="ols",
        trendline_color_override="#e74c3c",
        color_discrete_sequence=["#3498db"],
        opacity=0.65,
        title=f"Regression: {y_col} ~ {x_col}",
    )
    fig.update_traces(
        hovertemplate=f"<b>{x_col}:</b> %{{x}}<br><b>{y_col}:</b> %{{y}}<extra></extra>",
        selector=dict(mode="markers"),
    )

    # Pull OLS results and annotate R²
    try:
        results = px.get_trendline_results(fig)
        r2 = results.iloc[0]["px_fit_results"].rsquared
        fig.add_annotation(
            text=f"R² = {r2:.3f}",
            xref="paper", yref="paper",
            x=0.02, y=0.97,
            showarrow=False,
            font=dict(size=13, color="#e74c3c"),
            bgcolor="rgba(255,255,255,0.7)",
        )
    except Exception:
        pass

    _apply_theme(fig, f"Regression: {y_col} ~ {x_col}")
    return fig