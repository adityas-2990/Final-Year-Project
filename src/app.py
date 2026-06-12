"""
app.py
──────
Main Streamlit entry point.

Run with:
    streamlit run app.py

Module map
──────────
    app.py            ← you are here (orchestration only)
    sidebar.py        ← all sidebar UI widgets
    data_utils.py     ← pandas / sklearn data helpers (no Streamlit)
    visualizations.py ← Matplotlib / Seaborn figure builders (no Streamlit)
    clustering.py     ← KMeans elbow + cluster figures (no Streamlit)
    chat.py           ← Ollama chat helpers (no Streamlit)
    graph_analysis.py ← Vision model chart analysis
    rag.py            ← RAG backend logic
    rag_tab.py        ← RAG Chat UI
    evaluation.py     ← RAG Mathematical/Neural metrics backend
    eval_tab.py       ← RAG Evaluation UI
"""

import pandas as pd
import streamlit as st

# ── Local modules ─────────────────────────────────────────────────────────────
import data_utils as du
import visualizations as viz
import clustering as cl
import chat as ch
import sidebar as sb
import graph_analysis as ga

import rag_tab
import eval_tab
import nl2pandas_tab 

# ──────────────────────────────────────────────────────────────────────────────
#  Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Data Analysis App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────────────────────────────────────
#  Session-state initialisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _init_state(df_raw: pd.DataFrame) -> None:
    if "df_original" not in st.session_state:
        st.session_state.df_original = df_raw.copy()
    if "df" not in st.session_state:
        st.session_state.df = df_raw.copy()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "graph_context" not in st.session_state:
        st.session_state.graph_context = []


# ──────────────────────────────────────────────────────────────────────────────
#  Chat panel (right column)
# ──────────────────────────────────────────────────────────────────────────────

def _render_chat(container, df: pd.DataFrame, model: str) -> None:
    """Render the chat UI inside `container` (a st.container or column)."""
    with container:
        st.subheader("💬 Chat with your Data")

        # ── Conversation history ───────────────────────────────────────────────
        chat_box = st.container(height=420)
        with chat_box:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

        # ── User input ─────────────────────────────────────────────────────────
        user_input = st.chat_input("Ask anything about your data...")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with chat_box:
                with st.chat_message("user"):
                    st.write(user_input)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        reply = _call_ollama(user_input, df, model)
                        st.write(reply)
                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": reply}
                        )

        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat", key="clear_chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()


def _send_message(question: str, df: pd.DataFrame, model: str) -> None:
    st.session_state.chat_history.append({"role": "user", "content": question})
    reply = _call_ollama(question, df, model)
    st.session_state.chat_history.append({"role": "assistant", "content": reply})


def _call_ollama(question: str, df: pd.DataFrame, model: str) -> str:
    try:
        # Build extra context from all graph analyses done this session
        graph_ctx = st.session_state.get("graph_context", [])
        graph_ctx_str = ""
        if graph_ctx:
            graph_ctx_str = (
                "\n\nGRAPH ANALYSES (from charts the user has plotted this session):\n"
                + "\n\n---\n\n".join(graph_ctx)
            )
        return ch.chat_with_data(
            df,
            question,
            st.session_state.chat_history[:-1],
            model=model,
            extra_context=graph_ctx_str,
        )
    except Exception as exc:
        return (
            f"⚠️ Could not reach Ollama.\n\n"
            f"**Error:** `{exc}`\n\n"
            "Make sure Ollama is running in your system tray."
        )


def _analyse_and_show(fig) -> None:
    """
    Convert `fig` to an image, send it to the qwen2.5vl vision model for
    analysis, append the result to the chat history, and store it in
    graph_context so llama3.2 can reference it in follow-up questions.
    """
    with st.spinner("🔍 Analysing chart with qwen2.5vl…"):
        analysis = ga.analyse_figure(fig)
    # Show in chat panel
    st.session_state.chat_history.append(
        {"role": "assistant", "content": f"📊 **Chart Analysis**\n\n{analysis}"}
    )
    # Store for llama3.2 context
    st.session_state.graph_context.append(analysis)


# ──────────────────────────────────────────────────────────────────────────────
#  Main content (left column)
# ──────────────────────────────────────────────────────────────────────────────

def _render_main(main_col, df: pd.DataFrame) -> None:
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    all_cols = df.columns.tolist()

    with main_col:
        # ── Dataset overview ───────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🗂️ Your Dataset")
        st.dataframe(df.head(), use_container_width=True)
        c1, c2 = st.columns(2)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])

        # ── Basic statistics ───────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📋 Basic Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)

        # ── Missing values overview ────────────────────────────────────────────
        st.markdown("---")
        st.subheader("❓ Missing Values")
        missing_summary = du.missing_summary(df)
        if not missing_summary.empty:
            st.dataframe(missing_summary, use_container_width=True)
            st.write("**Missing Values Heatmap**")
            st.plotly_chart(viz.plot_missing_heatmap(df), use_container_width=True)
        else:
            st.success("✅ No missing values found!")

        # ── Missing-value controls (sidebar) ───────────────────────────────────
        missing_cols = df.columns[df.isnull().any()].tolist()
        mv = sb.render_missing_value_controls(missing_cols)

        if mv["apply"] and mv["column"]:
            st.session_state.df = du.fill_missing(
                st.session_state.df, mv["column"], mv["method"]
            )
            st.rerun()
        if mv["knn_apply"]:
            st.session_state.df = du.knn_impute(st.session_state.df, mv["knn_k"])
            st.rerun()
        if mv["reset"]:
            st.session_state.df = st.session_state.df_original.copy()
            st.rerun()

        # ── Outlier detection ──────────────────────────────────────────────────
        out = sb.render_outlier_controls(numerical_cols)
        if out["detect"] and out["column"]:
            st.markdown("---")
            st.subheader(f"🎯 Outliers in '{out['column']}' ({out['method'].upper()})")
            outliers = du.detect_outliers(df, out["column"], out["method"])
            st.write(f"**{len(outliers)} outliers found.**")
            st.dataframe(outliers, use_container_width=True)
            _boxplot_fig = viz.plot_boxplot(df, out["column"])
            st.plotly_chart(_boxplot_fig, use_container_width=True)
            _analyse_and_show(_boxplot_fig)

        # ── Visualisations ─────────────────────────────────────────────────────
        vis = sb.render_visualisation_controls(all_cols, numerical_cols)

        if vis["bar"]["plot"]:
            st.markdown("---")
            st.subheader("Bar Chart")
            _bar_fig = viz.plot_bar_chart(df, vis["bar"]["x"], vis["bar"]["y"])
            st.plotly_chart(_bar_fig, use_container_width=True)
            _analyse_and_show(_bar_fig)

        if vis["numeric"]["plot"]:
            st.markdown("---")
            st.subheader(vis["numeric"]["type"])
            _num_fig = viz.plot_numeric_graph(
                df, vis["numeric"]["x"], vis["numeric"]["y"], vis["numeric"]["type"]
            )
            st.plotly_chart(_num_fig, use_container_width=True)
            _analyse_and_show(_num_fig)

        if vis["corr"]["plot"]:
            st.markdown("---")
            st.subheader("🔗 Correlation Heatmap")
            _corr_fig = viz.plot_correlation_heatmap(df)
            if _corr_fig:
                st.plotly_chart(_corr_fig, use_container_width=True)
                _analyse_and_show(_corr_fig)
            else:
                st.warning("Need at least 2 numerical columns for a correlation heatmap.")

        if vis["pie"].get("plot"):
            st.markdown("---")
            st.subheader(f"{vis['pie']['style']} — {vis['pie']['column']}")
            _pie_fig = viz.plot_pie_chart(
                df,
                vis["pie"]["column"],
                vis["pie"]["style"],
                vis["pie"]["show_pct"],
            )
            st.plotly_chart(_pie_fig, use_container_width=True)
            _analyse_and_show(_pie_fig)

        if vis["hist"].get("plot"):
            h = vis["hist"]
            st.markdown("---")
            st.subheader(f"Distribution of '{h['column']}'")
            _hist_fig = viz.plot_histogram(
                df, h["column"], h["bins"],
                h["bar_color"], h["kde_color"],
                h["show_kde"], h["show_mean_median"],
            )
            st.plotly_chart(_hist_fig, use_container_width=True)
            _analyse_and_show(_hist_fig)
            skew = df[h["column"]].dropna().skew()
            label = du.skewness_label(skew)
            if label == "symmetric":
                st.info("✅ Distribution is approximately symmetric.")
            else:
                st.warning(f"⚠️ {label.capitalize()} distribution (skew={skew:.2f}).")

        # ── Clustering ─────────────────────────────────────────────────────────
        clust = sb.render_clustering_controls(numerical_cols)

        if clust["elbow"] and clust["x"] and clust["y"]:
            if clust["x"] == clust["y"]:
                st.warning("⚠️ Please select two **different** columns for X and Y axes.")
            else:
                st.markdown("---")
                st.subheader("📐 Elbow Method — Optimal Clusters")
                inertias = cl.compute_inertias(df, clust["x"], clust["y"])
                _elbow_fig = cl.plot_elbow(inertias)
                st.plotly_chart(_elbow_fig, use_container_width=True)
                _analyse_and_show(_elbow_fig)

        if clust["plot"] and clust["x"] and clust["y"]:
            if clust["x"] == clust["y"]:
                st.warning("⚠️ Please select two **different** columns for X and Y axes.")
            else:
                st.markdown("---")
                st.subheader(f"🔵 KMeans Clustering (k={clust['n_clusters']})")
                _cluster_fig = cl.fit_and_plot_clusters(df, clust["x"], clust["y"], clust["n_clusters"])
                st.plotly_chart(_cluster_fig, use_container_width=True)
                _analyse_and_show(_cluster_fig)

        # ── Regression ─────────────────────────────────────────────────────────
        reg = sb.render_regression_controls(numerical_cols)
        if reg["plot"] and reg["x"] and reg["y"]:
            st.markdown("---")
            st.subheader("📉 Regression Plot")
            _reg_fig = viz.plot_regression(df, reg["x"], reg["y"])
            st.plotly_chart(_reg_fig, use_container_width=True)
            _analyse_and_show(_reg_fig)

        # ── Download ───────────────────────────────────────────────────────────
        if sb.render_download_controls():
            st.markdown("---")
            st.subheader("Full Dataset")
            st.dataframe(df, use_container_width=True)
            st.markdown(du.df_to_csv_download_link(df), unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("📊 Data Analysis App")
    st.write(
        "Upload your CSV dataset to explore, clean, visualise, and cluster your data — "
        "then chat with it using a local AI model via Ollama."
    )

    # ── Sidebar: upload ───────────────────────────────────────────────────────
    uploaded_file = sb.render_upload_section()
    
    # Define a default model since the sidebar selector was removed
    ollama_model = "llama3.2:latest"
    st.session_state["ollama_model"] = ollama_model

    # ── Load CSV and init session state BEFORE tabs ───────────────────────────
    # This must happen here so that ALL tabs (including NL2Pandas) can access
    # st.session_state.df on the very first run, not just after tab_analysis
    # has had a chance to execute.
    if uploaded_file is not None:
        st.sidebar.success("✅ File uploaded!")
        st.sidebar.markdown("---")
        df_raw = pd.read_csv(uploaded_file)
        
        # Clean column names by stripping leading/trailing whitespace
        df_raw.columns = df_raw.columns.str.strip()
        
        _init_state(df_raw)

    # ── Top-level tabs ────────────────────────────────────────────────────────
    tab_analysis, tab_rag, tab_eval, tab_nl = st.tabs([
        "📊 Data Visualiser",
        "🗂️ Chat with Documents",
        "🧪 RAG Evaluation",
        "🧠 NL2Pandas"
    ])

    # ── Tab 3 : Evaluation ────────────────────────────────────────────────────
    with tab_eval:
        # Fetch the active document collection from session state if it exists
        current_collection = st.session_state.get("rag_collection", None)
        
        eval_tab.render(
            chat_model=ollama_model, 
            embed_model="nomic-embed-text", 
            collection=current_collection
        )

    # ── Tab 2 : RAG — Chat with Documents (no CSV required) ───────────────────
    with tab_rag:
        rag_tab.render(chat_model=ollama_model)

    with tab_nl:
        if "df" in st.session_state and st.session_state.df is not None:
            nl2pandas_tab.render(st.session_state.df, ollama_model)
        else:
            st.info("👈 Upload a CSV file from the sidebar to use NL2Pandas.")

    # ── Tab 1 : Data Analysis ──────────────────────────────────────────────────
    with tab_analysis:
        if uploaded_file is None:
            st.info("👈 Upload a CSV file from the sidebar to get started.")
            return

        # df is already initialised above before tabs; just read it here
        df: pd.DataFrame = st.session_state.df

        # ── Sticky chat column CSS ─────────────────────────────────────────────
        # Targets the second column in the data-analysis tab and makes it stick
        # to the viewport top as the user scrolls through plots in the first col.
        st.markdown(
            """
            <style>
            /* Pin the right (chat) column so it stays beside every graph */
            [data-testid="stHorizontalBlock"] > div:nth-child(2) {
                position: sticky;
                top: 3.5rem;          /* clear the Streamlit toolbar */
                align-self: flex-start;
                max-height: calc(100vh - 4rem);
                overflow-y: auto;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        main_col, chat_col = st.columns([2, 1], gap="large")
        _render_main(main_col, df)
        _render_chat(chat_col, df, ollama_model)


if __name__ == "__main__":
    main()