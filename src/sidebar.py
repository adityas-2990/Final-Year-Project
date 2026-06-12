"""
sidebar.py
──────────
All Streamlit sidebar controls.

Each function renders a sidebar section and returns the user's current
selections as a plain Python dict — no side-effects, no st.session_state
mutations.  Actions that *do* mutate state (e.g. "Apply imputation") are
handled via callback patterns so the caller (app.py) decides what to do.
"""

import streamlit as st


# ──────────────────────────────────────────────────────────────────────────────
#  File upload
# ──────────────────────────────────────────────────────────────────────────────

def render_upload_section():
    """Render the file-upload widget and return the UploadedFile or None."""
    st.sidebar.header("📂 Upload Data")
    uploaded = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    st.sidebar.markdown("---")
    return uploaded


# ──────────────────────────────────────────────────────────────────────────────
#  Missing-value imputation controls
# ──────────────────────────────────────────────────────────────────────────────

def render_missing_value_controls(missing_cols: list[str]) -> dict:
    """
    Render the 'Handle Missing Values' sidebar section.

    Returns a dict with keys:
        column   : selected column (str or None)
        method   : "mean" | "median" | "mode" | "drop"
        apply    : bool  — True when the user clicked Apply
        knn_k    : int   — number of KNN neighbours
        knn_apply: bool  — True when the user clicked Apply KNN Imputer
        reset    : bool  — True when the user clicked Reset Data
    """
    st.sidebar.header("🩹 Handle Missing Values")
    result = {
        "column": None,
        "method": "mean",
        "apply": False,
        "knn_k": 5,
        "knn_apply": False,
        "reset": False,
    }

    METHOD_LABELS = {
        "Fill with Mean": "mean",
        "Fill with Median": "median",
        "Fill with Mode": "mode",
        "Drop Column": "drop",
    }

    if missing_cols:
        result["column"] = st.sidebar.selectbox(
            "Select Column", options=missing_cols, key="mv_col"
        )
        label = st.sidebar.radio(
            f"Method for '{result['column']}'",
            options=list(METHOD_LABELS.keys()),
        )
        result["method"] = METHOD_LABELS[label]
        result["apply"] = st.sidebar.button("Apply", key="apply_mv")
    else:
        st.sidebar.success("No missing values!")

    st.sidebar.markdown("---")
    st.sidebar.header("🧠 KNN Imputer")
    result["knn_k"] = st.sidebar.slider(
        "Number of Neighbors", min_value=1, max_value=10, value=5
    )
    result["knn_apply"] = st.sidebar.button("Apply KNN Imputer", key="knn_impute")

    st.sidebar.markdown("---")
    result["reset"] = st.sidebar.button("🔄 Reset Data", key="reset_data")
    st.sidebar.markdown("---")
    return result


# ──────────────────────────────────────────────────────────────────────────────
#  Outlier detection controls
# ──────────────────────────────────────────────────────────────────────────────

def render_outlier_controls(numerical_cols: list[str]) -> dict:
    """
    Returns:
        column : str
        method : "zscore" | "iqr"
        detect : bool
    """
    st.sidebar.header("🎯 Outlier Detection")
    result = {"column": None, "method": "zscore", "detect": False}

    if not numerical_cols:
        st.sidebar.warning("No numerical columns available.")
        return result

    result["column"] = st.sidebar.selectbox(
        "Select Column", options=numerical_cols, key="out_col"
    )
    method_label = st.sidebar.radio(
        "Detection Method", options=["Z-Score", "IQR"]
    )
    result["method"] = "zscore" if method_label == "Z-Score" else "iqr"
    result["detect"] = st.sidebar.button("Detect Outliers", key="detect_outliers")
    st.sidebar.markdown("---")
    return result


# ──────────────────────────────────────────────────────────────────────────────
#  Visualisation controls
# ──────────────────────────────────────────────────────────────────────────────

def render_visualisation_controls(df_cols: list[str], numerical_cols: list[str]) -> dict:
    """
    Renders all chart control widgets.

    Returns a nested dict with sub-keys for each chart type.
    """
    st.sidebar.header("📈 Data Visualisation")
    result: dict = {}

    # ── Bar chart ──────────────────────────────
    st.sidebar.subheader("Bar Chart (Categorical)")
    cat_selected = st.sidebar.multiselect(
        "Select Categorical Columns", options=df_cols, key="cat_cols"
    )
    x_cat = st.sidebar.selectbox("X-axis (numeric)", options=numerical_cols, key="x_cat")
    y_cat_opts = cat_selected if cat_selected else ["—"]
    y_cat = st.sidebar.selectbox("Y-axis (category)", options=y_cat_opts, key="y_cat")
    plot_bar = st.sidebar.button("Plot Bar Chart", key="plot_bar")
    result["bar"] = {
        "x": x_cat,
        "y": y_cat,
        "plot": plot_bar and bool(cat_selected) and y_cat != "—",
    }

    # ── Numeric graph ──────────────────────────
    st.sidebar.subheader("Numeric Graph")
    x_num = st.sidebar.selectbox("X-axis", options=numerical_cols, key="x_num")
    y_num = st.sidebar.selectbox("Y-axis", options=numerical_cols, key="y_num")
    graph_type = st.sidebar.selectbox(
        "Graph Type", options=["Scatter Plot", "Line Plot", "Bar Plot"]
    )
    plot_num = st.sidebar.button("Plot Graph", key="plot_num")
    result["numeric"] = {
        "x": x_num, "y": y_num, "type": graph_type, "plot": plot_num
    }

    # ── Correlation heatmap ────────────────────
    st.sidebar.subheader("Correlation Heatmap")
    result["corr"] = {
        "plot": st.sidebar.button("Plot Correlation Heatmap", key="plot_corr")
    }

    st.sidebar.markdown("---")

    # ── Pie / Donut ────────────────────────────
    st.sidebar.header("🥧 Pie / Donut Chart")
    result["pie"] = {"plot": False}
    # Only columns with ≤15 unique values make sense for a pie chart
    pie_eligible = [c for c in df_cols if True]  # evaluated at render time
    if pie_eligible:
        pie_col = st.sidebar.selectbox("Select Column", options=pie_eligible, key="pie_col")
        chart_style = st.sidebar.radio("Style", options=["Pie Chart", "Donut Chart"])
        show_pct = st.sidebar.checkbox("Show Percentages", value=True)
        result["pie"] = {
            "column": pie_col,
            "style": chart_style,
            "show_pct": show_pct,
            "plot": st.sidebar.button("Plot Chart", key="plot_pie"),
        }
    else:
        st.sidebar.warning("No suitable columns (<=15 unique values) for pie chart.")

    st.sidebar.markdown("---")

    # ── Histogram ─────────────────────────────
    st.sidebar.header("📊 Histogram + KDE")
    result["hist"] = {"plot": False}
    if numerical_cols:
        hist_col = st.sidebar.selectbox("Column", options=numerical_cols, key="hist_col")
        bins = st.sidebar.slider("Bins", 5, 100, 30, 5)
        bar_color = st.sidebar.color_picker("Bar Colour", "#4C72B0")
        kde_color = st.sidebar.color_picker("KDE Colour", "#DD4444")
        show_kde = st.sidebar.checkbox("Show KDE", value=True)
        show_mm = st.sidebar.checkbox("Show Mean & Median", value=True)
        result["hist"] = {
            "column": hist_col,
            "bins": bins,
            "bar_color": bar_color,
            "kde_color": kde_color,
            "show_kde": show_kde,
            "show_mean_median": show_mm,
            "plot": st.sidebar.button("Plot Histogram", key="plot_hist"),
        }

    st.sidebar.markdown("---")
    return result


# ──────────────────────────────────────────────────────────────────────────────
#  Clustering controls
# ──────────────────────────────────────────────────────────────────────────────

def render_clustering_controls(numerical_cols: list[str]) -> dict:
    """
    Returns:
        x, y        : feature columns
        n_clusters  : int
        elbow       : bool
        plot        : bool
    """
    st.sidebar.header("🔵 Clustering (KMeans)")
    result = {"x": None, "y": None, "n_clusters": 3, "elbow": False, "plot": False}

    if not numerical_cols:
        st.sidebar.warning("No numerical columns for clustering.")
        return result

    result["x"] = st.sidebar.selectbox("X-axis feature", options=numerical_cols, key="cl_x")
    result["y"] = st.sidebar.selectbox("Y-axis feature", options=numerical_cols, key="cl_y")
    result["n_clusters"] = st.sidebar.slider(
        "Number of Clusters", min_value=2, max_value=10, value=3
    )
    result["elbow"] = st.sidebar.button("Show Elbow Chart", key="elbow")
    result["plot"] = st.sidebar.button("Plot Clusters", key="plot_clusters")
    st.sidebar.markdown("---")
    return result


# ──────────────────────────────────────────────────────────────────────────────
#  Regression controls
# ──────────────────────────────────────────────────────────────────────────────

def render_regression_controls(numerical_cols: list[str]) -> dict:
    """Returns: x, y, plot."""
    st.sidebar.header("📉 Regression Plot")
    result = {"x": None, "y": None, "plot": False}

    if len(numerical_cols) >= 2:
        result["x"] = st.sidebar.selectbox("X-axis", options=numerical_cols, key="x_reg")
        result["y"] = st.sidebar.selectbox("Y-axis", options=numerical_cols, key="y_reg")
        result["plot"] = st.sidebar.button("Plot Regression", key="plot_reg")
    st.sidebar.markdown("---")
    return result


# ──────────────────────────────────────────────────────────────────────────────
#  Download controls
# ──────────────────────────────────────────────────────────────────────────────

def render_download_controls() -> bool:
    """Render download button; returns True if clicked."""
    clicked = st.sidebar.button("📥 View & Download Dataset", key="download")
    st.sidebar.markdown("---")
    return clicked