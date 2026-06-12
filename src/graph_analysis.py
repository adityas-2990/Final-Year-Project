"""
graph_analysis.py
─────────────────
Converts a Plotly figure to a PNG image and sends it to qwen3-vl:235b-cloud
via the Ollama Cloud API for vision-based analysis.

Requires OLLAMA_API_KEY to be set in a .env file in the project root.
Create your key at: https://ollama.com

Dependencies
────────────
    ollama       pip install ollama
    kaleido      pip install kaleido   (Plotly → PNG export)
    python-dotenv pip install python-dotenv
"""

from __future__ import annotations

import base64
import os

import plotly.graph_objects as go
from ollama import Client
from dotenv import load_dotenv

# Load environment variables from .env file in the project root
load_dotenv()

VISION_MODEL: str = "qwen3-vl:235b-cloud"

VISION_PROMPT = """\
You are an expert data analyst. Analyse the chart shown in the image carefully.

Respond in exactly this format — no extra text before or after:

**Summary**
<2-3 sentences describing what the chart shows overall>

**Insights**
1. <specific, data-driven insight>
2. <specific, data-driven insight>
3. <specific, data-driven insight>
4. <specific, data-driven insight — only if clearly supported by the data>
5. <specific, data-driven insight — only if clearly supported by the data>

Only include insights 4 and 5 if the chart genuinely supports them.
"""


def _get_client() -> Client:
    """Build an Ollama Client pointed at the cloud API with auth."""
    api_key = os.environ.get("OLLAMA_API_KEY", "")
    return Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {api_key}"},
    )


def _fig_to_base64(fig: go.Figure) -> str:
    """Render a Plotly figure to PNG bytes and return as a base64 string."""
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")
    png_bytes: bytes = fig.to_image(format="png", width=900, height=500, scale=1.5)
    return base64.standard_b64encode(png_bytes).decode("utf-8")


def analyse_figure(fig: go.Figure) -> str:
    """
    Send the figure to qwen3-vl:235b-cloud via the Ollama Cloud API and
    return the formatted analysis string (summary + 3-5 insights).

    Returns an error message string on failure so the caller can display
    it gracefully without crashing.
    """
    api_key = os.environ.get("OLLAMA_API_KEY")

    if not api_key:
        return (
            "⚠️ OLLAMA_API_KEY is not set.\n\n"
            "Add it to your .env file in the project root:\n"
            "`OLLAMA_API_KEY=your_api_key`"
        )

    try:
        image_b64 = _fig_to_base64(fig)
    except Exception as exc:
        return (
            f"⚠️ Could not render chart image: {exc}\n\n"
            "Make sure `kaleido` is installed: `pip install kaleido`"
        )

    try:
        client = _get_client()
        response = client.chat(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": VISION_PROMPT,
                    "images": [image_b64],
                }
            ],
        )
        return response.message.content
    except Exception as exc:
        return (
            f"⚠️ Ollama Cloud error: {exc}\n\n"
            "Check that your OLLAMA_API_KEY is valid and you have an active internet connection."
        )