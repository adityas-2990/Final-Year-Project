"""
nl2pandas_tab.py
────────────────
Streamlit UI for the NL2Pandas Query Engine.

Drop into app.py:

    tab_analysis, tab_rag, tab_eval, tab_nl = st.tabs([...])
    with tab_nl:
        import nl2pandas_tab
        nl2pandas_tab.render(st.session_state.df, ollama_model)

No query logic lives here — all pipeline code is in nl2pandas.py.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import nl2pandas as nl


# ──────────────────────────────────────────────────────────────────────────────
#  Session state
# ──────────────────────────────────────────────────────────────────────────────

_KEY_HISTORY = "nl2pandas_history"


def _init_state() -> None:
    if _KEY_HISTORY not in st.session_state:
        st.session_state[_KEY_HISTORY] = []


# ──────────────────────────────────────────────────────────────────────────────
#  Example queries (shown as clickable chips)
# ──────────────────────────────────────────────────────────────────────────────

_EXAMPLES = [
    "📋 Show the first 10 rows",
    "❓ How many missing values does each column have?",
    "📊 Count of records per category column",
    "🔝 Top 10 rows sorted by the largest numeric column descending",
    "📈 Average of each numeric column",
    "🔍 Show rows where any value is missing",
]


# ──────────────────────────────────────────────────────────────────────────────
#  Result renderer
# ──────────────────────────────────────────────────────────────────────────────

def _render_result(entry: dict, df_live: pd.DataFrame, idx: int) -> None:
    """Render a single history entry (query + result)."""

    # ── Query bubble ──────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="
            background: #f0f4ff;
            border-left: 4px solid #4c72b0;
            padding: 0.6rem 1rem;
            border-radius: 6px;
            margin-bottom: 0.4rem;
            font-size: 0.95rem;
            color: #1f2937;
        ">
            🧑 <strong>{entry['query']}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Generated code (collapsed by default) ────────────────────────────────
    if entry.get("code"):
        with st.expander("🔍 View generated pandas code", expanded=False):
            st.code(entry["code"], language="python")

            # Show safety stage badge
            stage = entry.get("stage", "ok")
            if stage == "ok":
                st.success("✅ Passed all 6 safety layers (AST validation + sandboxed execution)")
            else:
                st.error(f"🚫 Blocked at stage: **{stage}**")

    # ── Result or error ───────────────────────────────────────────────────────
    if not entry["success"]:
        st.error(f"⚠️ {entry['error']}")
        return

    result      = entry["result"]
    result_type = entry["result_type"]

    if result_type == "dataframe":
        n_rows, n_cols = result.shape
        st.markdown(
            f"<span style='color:#888; font-size:0.85rem'>"
            f"Returned **{n_rows}** row(s) × **{n_cols}** column(s)</span>",
            unsafe_allow_html=True,
        )
        st.dataframe(result, use_container_width=True)

        # "Apply as active dataset" button — the actionable part
        if st.button(
            "⚡ Apply as active dataset",
            key=f"nl_apply_{idx}",
            help="Replace the current working DataFrame with this filtered result.",
        ):
            st.session_state.df = result.reset_index(drop=True)
            st.success(
                f"✅ Active dataset updated to {n_rows} rows. "
                "Switch to the Data Visualiser tab to explore it."
            )

    elif result_type == "series":
        st.dataframe(result.to_frame(), use_container_width=True)

    else:
        # Scalar — display as a big metric
        st.metric(label="Result", value=str(round(result, 6) if isinstance(result, float) else result))

    st.markdown("---")


# ──────────────────────────────────────────────────────────────────────────────
#  Core render function
# ──────────────────────────────────────────────────────────────────────────────

def render(df: pd.DataFrame, model: str = "llama3.2:latest") -> None:
    """
    Render the NL2Pandas tab. Call from app.py inside `with tab_nl:`.

    Parameters
    ----------
    df    : The active working DataFrame from session state.
    model : The Ollama model tag to use for code generation.
    """
    _init_state()

    st.subheader("🧠 NL2Pandas — Query your Data in Plain English")
    st.caption(
        "Type a question in natural language. The AI will write and safely "
        "execute the pandas code for you — no coding required."
    )

    # ── Guard: dataset must be loaded ────────────────────────────────────────
    if df is None or df.empty:
        st.info("👈 Upload a CSV file from the sidebar first to use this feature.")
        return

    # ── Dataset schema reference ──────────────────────────────────────────────
    with st.expander("📋 Active dataset schema", expanded=False):
        schema_df = pd.DataFrame({
            "Column": df.dtypes.index,
            "Dtype":  df.dtypes.values.astype(str),
            "Non-Null": df.notnull().sum().values,
            "Sample": [str(df[c].dropna().iloc[0]) if df[c].notnull().any() else "—"
                    for c in df.columns],
        })
        st.dataframe(schema_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Example query chips ───────────────────────────────────────────────────
    st.markdown("**✨ Try an example query:**")
    cols = st.columns(3)
    for i, example in enumerate(_EXAMPLES):
        if cols[i % 3].button(example, key=f"nl_ex_{i}", use_container_width=True):
            _execute_and_store(example, df, model)
            st.rerun()

    st.markdown("")

    # ── Main query input ──────────────────────────────────────────────────────
    user_query = st.chat_input(
        "Ask anything about your data… e.g. 'show rows where age > 30 and city is Mumbai'"
    )

    if user_query:
        _execute_and_store(user_query, df, model)
        st.rerun()

    # ── History ───────────────────────────────────────────────────────────────
    history = st.session_state[_KEY_HISTORY]

    if not history:
        st.markdown(
            """
            <div style="text-align:center; padding: 3rem 1rem; color:#aaa;">
                <div style="font-size:3rem;">🔎</div>
                <p style="font-size:1.1rem; margin-top:0.5rem;">
                    Your query results will appear here.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Render newest first
    for i, entry in enumerate(reversed(history)):
        _render_result(entry, df, len(history) - 1 - i)

    # Clear history
    if st.button("🗑️ Clear Query History", use_container_width=True, key="nl_clear"):
        st.session_state[_KEY_HISTORY] = []
        st.rerun()

    # ── Safety layer explainer (thesis-facing) ────────────────────────────────
    with st.expander("🔐 How the safety sandbox works", expanded=False):
        st.markdown("""
        Every piece of generated code passes through **6 security layers** before execution:

        | Layer | Mechanism | Blocks |
        |-------|-----------|--------|
        | 1 | AST node type filter | `import`, `from X import`, `delete` |
        | 2 | Forbidden name check | `exec`, `eval`, `os`, `sys`, `open`, `subprocess`… |
        | 3 | Dunder attribute guard | `__class__`, `__bases__`, `__subclasses__`… |
        | 4 | Isolated namespace | No builtins injected (`__builtins__ = {}`) |
        | 5 | Copy-on-execute | Original DataFrame is never mutated |
        | 6 | Execution timeout | Queries killed after 8s (prevents infinite loops) |

        Even if the LLM were to generate malicious code (prompt injection), 
        layers 1–3 catch it at parse time before a single line runs.
        """)


# ──────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _execute_and_store(
    query: str,
    df: pd.DataFrame,
    model: str,
) -> None:
    """Run the full NL2Pandas pipeline and append the result to history."""
    with st.spinner("🤖 Generating and executing pandas code…"):
        result = nl.run_query(df, query, model)

    st.session_state[_KEY_HISTORY].append({
        "query": query,
        **result,
    })