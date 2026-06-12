"""
chat.py
───────
Ollama-powered chat helpers.

Responsibilities
────────────────
- Build a dataset context string from a DataFrame.
- Send a question + conversation history to Ollama and return the reply.
- Define the suggested quick-questions list.

No Streamlit UI code lives here — rendering is handled in sidebar.py.
"""

import pandas as pd
import ollama


# ──────────────────────────────────────────────────────────────────────────────
#  Suggested quick questions
# ──────────────────────────────────────────────────────────────────────────────

SUGGESTED_QUESTIONS: list[str] = [
    "📋 Give me a summary of this dataset",
    "🔍 Which columns have missing data and what should I do?",
    "📈 What are the most interesting patterns or trends?",
    "⚠️ Are there any potential data quality issues?",
    "🔗 Which features are likely to be correlated?",
    "🤔 What kind of ML models would suit this dataset?",
]


# ──────────────────────────────────────────────────────────────────────────────
#  Dataset context builder
# ──────────────────────────────────────────────────────────────────────────────

def build_dataset_context(df: pd.DataFrame) -> str:
    """Return a compact text summary of a DataFrame for the LLM system prompt."""
    missing = df.isnull().sum()
    missing_str = (
        missing[missing > 0].to_string() if missing.any() else "None"
    )
    return (
        f"Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns\n\n"
        f"Columns & Data Types:\n{df.dtypes.to_string()}\n\n"
        f"Statistical Summary:\n{df.describe().to_string()}\n\n"
        f"First 5 Rows:\n{df.head().to_string()}\n\n"
        f"Missing Values:\n{missing_str}"
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Ollama chat
# ──────────────────────────────────────────────────────────────────────────────

def chat_with_data(
    df: pd.DataFrame,
    user_question: str,
    history: list[dict],
    model: str = "llama3.2:latest",
    extra_context: str = "",
) -> str:
    """
    Send `user_question` to Ollama together with `history` and the
    dataset context embedded in the system prompt.

    Parameters
    ----------
    df            : DataFrame the user has uploaded.
    user_question : The new user message.
    history       : Previous turns as [{"role": ..., "content": ...}, ...].
                    Should NOT include the current user_question.
    model         : Ollama model tag.
    extra_context : Optional additional context (e.g. graph analyses from
                    qwen2.5vl) appended to the system prompt so the model
                    can answer questions about plotted charts.

    Returns
    -------
    Assistant reply as a plain string.
    """
    system_prompt = (
        "You are an expert data analyst assistant. "
        "The user has uploaded a dataset whose summary is provided below. "
        "Answer questions clearly and concisely, referencing specific columns, "
        "statistics, or patterns from the data wherever relevant. "
        "If the user asks about a chart or graph, use the graph analyses "
        "provided below to give accurate, specific answers."
        "\n\n"
        f"DATASET SUMMARY:\n{build_dataset_context(df)}"
        + extra_context
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_question})

    response = ollama.chat(model=model, messages=messages)
    return response["message"]["content"]


# ──────────────────────────────────────────────────────────────────────────────
#  Startup health-check
# ──────────────────────────────────────────────────────────────────────────────

def ollama_available_models() -> list[str] | None:
    """
    Return a list of locally-pulled model names, or None if Ollama
    is unreachable (e.g. the daemon is not running).
    """
    try:
        return [m["name"] for m in ollama.list()["models"]]
    except Exception:
        return None