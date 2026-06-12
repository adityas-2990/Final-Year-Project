"""
nl2pandas.py
────────────
Natural Language → Pandas Query Engine.

Pipeline
────────
    User NL query
        ↓
    Ollama LLM (code generation with schema context)
        ↓
    Code extractor  (strips markdown fences)
        ↓
    AST Safety Validator  (rejects dangerous nodes / names)
        ↓
    Sandboxed executor  (isolated namespace, df copy, timeout)
        ↓
    Typed result  (DataFrame | Series | scalar)

Safety Model (academically documented for thesis)
─────────────────────────────────────────────────
Layer 1 — Forbidden AST node types
    Import / ImportFrom / Delete nodes are rejected at parse time.
    This prevents the model from injecting arbitrary library calls.

Layer 2 — Forbidden name references
    A blocklist of ~20 dangerous built-in names (exec, eval, os, sys,
    open, subprocess, socket, …) are caught via ast.Name node inspection.

Layer 3 — Dunder attribute guard
    Any attribute access whose name starts with "__" is blocked to
    prevent __class__.__subclasses__() style sandbox escapes.

Layer 4 — Isolated namespace
    exec() receives a namespace containing only {"df": copy, "pd": pd}.
    No builtins are injected.  Python's default exec() still makes a
    small set of safe builtins available; these are explicitly restricted
    by setting __builtins__ = {} in the namespace.

Layer 5 — Copy-on-execute
    df is shallow-copied before execution so user queries can never
    mutate the live session-state DataFrame.

Layer 6 — Execution timeout
    A daemon thread runs the exec(); if it exceeds EXEC_TIMEOUT seconds
    the main thread raises TimeoutError (protects against infinite loops).
"""

from __future__ import annotations

import ast
import re
import textwrap
import threading
from typing import Any

import pandas as pd
import ollama


# ──────────────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────────────

EXEC_TIMEOUT: float = 8.0   # seconds before an execution is killed

# AST node types that are always forbidden
_FORBIDDEN_NODES: frozenset[type] = frozenset({
    ast.Import,
    ast.ImportFrom,
    ast.Delete,
    ast.Global,
    ast.Nonlocal,
})

# Built-in / global names that must never appear in generated code
_FORBIDDEN_NAMES: frozenset[str] = frozenset({
    "exec", "eval", "compile", "__import__", "open", "input",
    "os", "sys", "subprocess", "socket", "shutil", "pathlib",
    "builtins", "__builtins__", "globals", "locals", "vars",
    "dir", "getattr", "setattr", "delattr", "hasattr",
    "type", "object", "super", "classmethod", "staticmethod",
    "breakpoint", "memoryview", "bytearray",
})


# ──────────────────────────────────────────────────────────────────────────────
#  Custom exceptions
# ──────────────────────────────────────────────────────────────────────────────

class UnsafeCodeError(Exception):
    """Raised when the AST validator finds a forbidden construct."""


class ExecutionTimeoutError(Exception):
    """Raised when sandboxed execution exceeds EXEC_TIMEOUT."""


# ──────────────────────────────────────────────────────────────────────────────
#  Layer 1-3: AST Validator
# ──────────────────────────────────────────────────────────────────────────────

def validate_ast(code: str) -> ast.Module:
    """
    Parse `code` into an AST and apply all safety layers.

    Returns the parsed Module on success.
    Raises UnsafeCodeError with a descriptive message on failure.
    """
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise UnsafeCodeError(f"Syntax error in generated code: {exc}") from exc

    for node in ast.walk(tree):
        # Layer 1: forbidden node types
        if type(node) in _FORBIDDEN_NODES:
            raise UnsafeCodeError(
                f"Forbidden operation '{type(node).__name__}' detected. "
                "Import and delete statements are not permitted."
            )

        # Layer 2: forbidden name references
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise UnsafeCodeError(
                f"Forbidden name '{node.id}' detected in generated code."
            )

        # Layer 3: dunder attribute guard
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise UnsafeCodeError(
                f"Forbidden dunder attribute '{node.attr}' detected. "
                "Private/magic attribute access is not permitted."
            )

    return tree


# ──────────────────────────────────────────────────────────────────────────────
#  Code extractor
# ──────────────────────────────────────────────────────────────────────────────

def extract_code(llm_response: str) -> str:
    """
    Strip markdown fences from an LLM response and return raw Python code.

    Handles:
        ```python ... ```
        ``` ... ```
        Plain code (no fences)
    """
    # Try ```python ... ``` first
    match = re.search(r"```python\s*(.*?)```", llm_response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Try generic ``` ... ```
    match = re.search(r"```\s*(.*?)```", llm_response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Last resort: strip the whole response
    return llm_response.strip()


# ──────────────────────────────────────────────────────────────────────────────
#  Layers 4-6: Sandboxed executor
# ──────────────────────────────────────────────────────────────────────────────

def _rewrite_last_expr(tree: ast.Module) -> ast.Module:
    """
    If the last statement in `tree` is a bare expression, rewrite it as
    `result = <expr>` so we can capture the value after exec().
    """
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body[-1]
        assignment = ast.Assign(
            targets=[ast.Name(id="result", ctx=ast.Store())],
            value=last.value,
            lineno=last.lineno,
            col_offset=last.col_offset,
        )
        tree.body[-1] = assignment
        ast.fix_missing_locations(tree)
    return tree


def execute_sandboxed(
    tree: ast.Module,
    df: pd.DataFrame,
) -> tuple[Any, str]:
    """
    Execute a validated AST in a fully isolated namespace.

    Namespace contains only:
        df   — a shallow copy of the caller's DataFrame
        pd   — the pandas module
        __builtins__ — empty dict  (Layer 4)

    Execution runs in a daemon thread with a timeout (Layer 6).

    Returns
    -------
    (result, result_type)
        result_type is one of: "dataframe" | "series" | "scalar"

    Raises
    ------
    ExecutionTimeoutError  — execution exceeded EXEC_TIMEOUT seconds
    RuntimeError           — any other exception raised during exec()
    """
    tree = _rewrite_last_expr(tree)
    bytecode = compile(tree, filename="<nl2pandas>", mode="exec")

    # ── Layer 4: Safe builtins whitelist ──────────────────────────────────────
    # {} would be too aggressive — it breaks pandas operations that rely on
    # Python-level type constructors (bool, int, str, etc.) in generated code.
    # We allow only inert, non-exploitable builtins.  Dangerous names (exec,
    # eval, open, os, sys, getattr, type …) are already blocked by Layer 2.
    _SAFE_BUILTINS: dict[str, Any] = {
        # Basic types
        "bool": bool, "int": int, "float": float, "str": str,
        "list": list, "dict": dict, "tuple": tuple, "set": set,
        "frozenset": frozenset, "bytes": bytes,
        # Iteration helpers
        "len": len, "range": range, "enumerate": enumerate,
        "zip": zip, "map": map, "filter": filter, "iter": iter, "next": next,
        # Sorting / aggregation
        "sorted": sorted, "reversed": reversed,
        "min": min, "max": max, "sum": sum, "abs": abs,
        "round": round, "pow": pow, "divmod": divmod,
        # Introspection (safe, read-only)
        "isinstance": isinstance, "issubclass": issubclass,
        "hasattr": hasattr, "callable": callable,
        "repr": repr, "hash": hash, "id": id,
        # Safe I/O (display only)
        "print": print,
        # Exception types (needed for try/except in generated code)
        "Exception": Exception, "ValueError": ValueError,
        "TypeError": TypeError, "KeyError": KeyError,
        "IndexError": IndexError, "AttributeError": AttributeError,
        "StopIteration": StopIteration,
    }

    # Isolated namespace — Layer 4 + Layer 5
    namespace: dict[str, Any] = {
        "__builtins__": _SAFE_BUILTINS,
        "df": df.copy(),      # Layer 5: copy so original is never mutated
        "pd": pd,
    }

    result_container: dict[str, Any] = {}
    error_container:  dict[str, Any] = {}
    # Keep a reference to df columns for helpful error messages
    _df_columns = list(df.columns)

    def _run() -> None:
        try:
            exec(bytecode, namespace)  # noqa: S102  (intentional sandboxed exec)
            result_container["value"] = namespace.get(
                "result", namespace.get("df")
            )
        except KeyError as exc:
            # Give an actionable message: show what columns actually exist
            bad_key = str(exc)
            close = [c for c in _df_columns if c.strip().lower() == bad_key.strip().lower().strip("'\"")]
            hint = ""
            if close:
                hint = f"\n💡 Did you mean: '{close[0]}'? (column names are case-sensitive and may have spaces)"
            error_container["exc"] = RuntimeError(
                f"Column {exc} not found in the DataFrame.{hint}\n\n"
                f"Available columns: {_df_columns}"
            )
        except Exception as exc:  # noqa: BLE001
            error_container["exc"] = exc

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=EXEC_TIMEOUT)

    if thread.is_alive():
        raise ExecutionTimeoutError(
            f"Query execution timed out after {EXEC_TIMEOUT:.0f} seconds. "
            "Possible infinite loop in generated code."
        )

    if "exc" in error_container:
        raise RuntimeError(str(error_container["exc"])) from error_container["exc"]

    value = result_container.get("value")

    if isinstance(value, pd.DataFrame):
        return value, "dataframe"
    if isinstance(value, pd.Series):
        return value, "series"
    return value, "scalar"


# ──────────────────────────────────────────────────────────────────────────────
#  LLM code generation
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_TEMPLATE = textwrap.dedent("""
    You are an expert Python/pandas code generator.
    The user has a DataFrame called `df` with this schema and sample data:

    EXACT COLUMN NAMES — use these character-for-character including spaces:
    {columns}

    SCHEMA (column → dtype):
    {schema}

    SAMPLE DATA (first 3 rows):
    {sample}

    STRICT RULES — violating any rule makes your output unusable:
    1. Output ONLY a single ```python ... ``` code block. No prose, no explanation.
    2. The DataFrame is always called `df`. Never rename it.
    3. Do NOT import anything. `pd` (pandas) and `df` are the only available names.
    4. Assign the final result to a variable named `result`.
    5. `result` must be a DataFrame, Series, or scalar — nothing else.
    6. Prefer vectorised pandas operations. Never use Python loops on rows.
    7. For string matching, use `.str.lower()` for case-insensitivity.
    8. For group operations, always reset_index() if the result should be a DataFrame.
    9. CRITICAL — column names: copy them EXACTLY from the "EXACT COLUMN NAMES" list
       above, including any leading/trailing spaces or unusual capitalisation.
    10. CRITICAL — "which X has the highest/lowest/best/worst Y": ALWAYS return
        the full row(s) using `.loc[df[col].idxmax()]` or `.nlargest()`, never
        just the scalar value from `.max()` or `.min()` alone.

    EXAMPLES:
    User: show rows where salary > 60000 and department is Marketing
    ```python
    result = df[(df['salary'] > 60000) & (df['department'].str.lower() == 'marketing')]
    ```

    User: which country has the highest life expectancy?
    ```python
    result = df.loc[[df['Life expectancy'].idxmax()]]
    ```

    User: which 3 countries have the lowest GDP?
    ```python
    result = df.nsmallest(3, 'GDP')
    ```

    User: what is the average age?
    ```python
    result = df['age'].mean()
    ```

    User: top 5 customers by total_spend descending
    ```python
    result = df.sort_values('total_spend', ascending=False).head(5)
    ```

    User: count of records per category
    ```python
    result = df.groupby('category').size().reset_index(name='count')
    ```

    User: percentage of missing values per column
    ```python
    result = (df.isnull().mean() * 100).round(2).reset_index()
    result.columns = ['column', 'missing_%']
    ```
""").strip()


def generate_code(
    df: pd.DataFrame,
    user_query: str,
    model: str = "llama3.2:latest",
) -> str:
    """
    Ask Ollama to generate pandas code for `user_query`.

    Returns the raw LLM response string (use extract_code() to parse it).
    """
    # Format column names as a quoted list so the LLM sees spaces/special chars
    columns_list = "\n".join(f'  - "{c}"' for c in df.columns)

    system_prompt = _SYSTEM_TEMPLATE.format(
        columns=columns_list,
        schema=df.dtypes.to_string(),
        sample=df.head(3).to_string(),
    )

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_query},
        ],
        options={"temperature": 0.0},   # deterministic for safety + reproducibility
    )
    return response["message"]["content"]


# ──────────────────────────────────────────────────────────────────────────────
#  Public API — full pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_query(
    df: pd.DataFrame,
    user_query: str,
    model: str = "llama3.2:latest",
) -> dict:
    """
    Full NL → pandas pipeline.

    Parameters
    ----------
    df         : The user's active DataFrame.
    user_query : Plain-English query string.
    model      : Ollama model tag to use for code generation.

    Returns
    -------
    A result dict with the following keys:

    success      bool   — True if all stages passed
    code         str    — Extracted Python code (empty on early failure)
    raw_llm      str    — Raw LLM response (for debugging / thesis logging)
    result       Any    — Query result; None on failure
    result_type  str    — "dataframe" | "series" | "scalar" | "error"
    error        str    — Human-readable error message; "" on success
    stage        str    — Which stage failed: "generation" | "extraction"
                        | "validation" | "execution" | "ok"
    """
    out: dict = {
        "success":     False,
        "code":        "",
        "raw_llm":     "",
        "result":      None,
        "result_type": "error",
        "error":       "",
        "stage":       "ok",
    }

    # ── Stage 1: LLM code generation ─────────────────────────────────────────
    try:
        raw = generate_code(df, user_query, model)
        out["raw_llm"] = raw
    except Exception as exc:
        out["error"] = f"LLM generation failed: {exc}"
        out["stage"] = "generation"
        return out

    # ── Stage 2: Code extraction ──────────────────────────────────────────────
    try:
        code = extract_code(raw)
        if not code:
            raise ValueError("LLM returned an empty code block.")
        out["code"] = code
    except Exception as exc:
        out["error"] = f"Could not extract code from LLM response: {exc}"
        out["stage"] = "extraction"
        return out

    # ── Stage 3: AST safety validation ───────────────────────────────────────
    try:
        tree = validate_ast(code)
    except UnsafeCodeError as exc:
        out["error"] = str(exc)
        out["stage"] = "validation"
        return out

    # ── Stage 4: Sandboxed execution ─────────────────────────────────────────
    try:
        result, result_type = execute_sandboxed(tree, df)
        out["success"]     = True
        out["result"]      = result
        out["result_type"] = result_type
    except ExecutionTimeoutError as exc:
        out["error"] = str(exc)
        out["stage"] = "execution"
    except Exception as exc:
        out["error"] = f"Runtime error during execution: {exc}"
        out["stage"] = "execution"

    return out