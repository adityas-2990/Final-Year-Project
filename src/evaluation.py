"""
evaluation.py
─────────────
Mathematical, statistical, and neural evaluation metrics for RAG pipelines.
"""

import re
import numpy as np
import ollama

# ── General Utilities ─────────────────────────────────────────────────────────

STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", 
    "of", "with", "is", "are", "was", "were", "it", "this", "that", "i", 
    "you", "they", "we", "he", "she", "from", "by", "as"
}

def _get_meaningful_words(text: str) -> set:
    """Extract lowercase alphanumeric words, stripping out common stop words."""
    words = set(re.findall(r'\b\w+\b', str(text).lower()))
    return set(words) - STOP_WORDS


# ── 1. Deterministic Text Metrics ─────────────────────────────────────────────

def calculate_token_f1(ai_answer: str, ground_truth: str) -> float:
    """Standard statistical F1 score based on token overlap."""
    ai_tokens = set(str(ai_answer).lower().split())
    truth_tokens = set(str(ground_truth).lower().split())
    
    common = ai_tokens.intersection(truth_tokens)
    if not common:
        return 0.0
        
    precision = len(common) / len(ai_tokens)
    recall = len(common) / len(truth_tokens)
    
    return round(2 * (precision * recall) / (precision + recall), 4)


# ── 2. Custom Metrics (Thesis Specific) ───────────────────────────────────────

def non_trivial_jaccard(ai_answer: str, ground_truth: str) -> float:
    """NTJS: Jaccard similarity ignoring common English stop words."""
    ai_set = _get_meaningful_words(ai_answer)
    truth_set = _get_meaningful_words(ground_truth)
    
    if not ai_set and not truth_set:
        return 1.0
    if not ai_set or not truth_set:
        return 0.0
        
    intersection = len(ai_set.intersection(truth_set))
    union = len(ai_set.union(truth_set))
    
    return round(intersection / union, 4)


def deterministic_grounding_index(ai_answer: str, retrieved_context: str) -> float:
    """DGI: Percentage of meaningful AI words found directly in the source text."""
    ai_words = _get_meaningful_words(ai_answer)
    context_words = _get_meaningful_words(retrieved_context)
    
    if not ai_words:
        return 0.0
        
    grounded_words = ai_words.intersection(context_words)
    return round(len(grounded_words) / len(ai_words), 4)


# ── 3. Vector, Neural & LLM Metrics ───────────────────────────────────────────

def cosine_similarity(ai_answer: str, ground_truth: str, embed_model: str) -> float:
    """Deterministic semantic similarity using vector dot product."""
    try:
        vec_a = ollama.embeddings(model=embed_model, prompt=ai_answer)["embedding"]
        vec_b = ollama.embeddings(model=embed_model, prompt=ground_truth)["embedding"]
        
        A = np.array(vec_a)
        B = np.array(vec_b)
        
        dot_product = np.dot(A, B)
        norm_a = np.linalg.norm(A)
        norm_b = np.linalg.norm(B)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return round(float(dot_product / (norm_a * norm_b)), 4)
    except Exception:
        return 0.0


def calculate_bertscore(ai_answer: str, ground_truth: str) -> float:
    """
    SOTA neural metric using contextual embeddings.
    Requires: pip install bert-score
    """
    try:
        from bert_score import score
        # Using a small, fast model to keep local evaluation quick
        P, R, F1 = score([ai_answer], [ground_truth], lang="en", model_type="distilbert-base-uncased")
        return round(float(F1.mean()), 4)
    except ImportError:
        # Fails gracefully if the user hasn't installed bert-score
        return -1.0 


def g_eval_score(question: str, ai_answer: str, ground_truth: str, chat_model: str) -> int:
    """LLM-as-a-judge: Rates response completeness from 1 to 5."""
    prompt = f"""
    You are an impartial grading system evaluating an AI's answer.
    Question: {question}
    Ground Truth: {ground_truth}
    AI Answer: {ai_answer}
    
    Rate the AI's answer from 1 to 5 based on accuracy, factual correctness, and completeness compared to the ground truth.
    Respond with ONLY a single integer (1, 2, 3, 4, or 5). Do not include any other text, reasoning, or punctuation.
    """
    try:
        response = ollama.chat(
            model=chat_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0}
        )
        match = re.search(r'\d', response["message"]["content"])
        return int(match.group()) if match else 0
    except Exception:
        return 0


# ── 4. Retrieval Metrics (Separated for clarity) ──────────────────────────────

def evaluate_retrieval(retrieved_chunks: list, expected_keywords: list) -> dict:
    """
    Returns Rank, Recall@K, and MRR.
    A chunk is considered the 'true' chunk if it contains at least 50% of the keywords.
    """
    result = {"Rank": "Not Found", "Recall@K": 0, "MRR": 0.0}
    
    if not expected_keywords or not retrieved_chunks:
        return result
        
    expected_lower = [str(kw).lower() for kw in expected_keywords]
    
    for index, chunk in enumerate(retrieved_chunks):
        chunk_lower = chunk.lower()
        hits = sum(1 for kw in expected_lower if kw in chunk_lower)
        
        if len(expected_lower) > 0 and (hits / len(expected_lower)) >= 0.5:
            result["Rank"] = index + 1
            result["Recall@K"] = 1
            result["MRR"] = round(1.0 / (index + 1), 4)
            return result # Return on first valid hit
            
    return result