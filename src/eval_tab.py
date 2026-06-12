"""
eval_tab.py
───────────
Streamlit UI for the RAG Evaluation Pipeline.
Upload a source document and a JSON dataset, tune chunking/retrieval parameters, 
and view the statistical dashboard updating in real-time.
"""

import json
import pandas as pd
import streamlit as st

import rag as rg
import evaluation as ev

def render(chat_model: str, embed_model: str, collection=None) -> None:
    """Render the RAG Evaluation tab."""
    st.subheader("🧪 RAG Hyperparameter & Evaluation Workbench")
    st.caption(
        "Upload your source document and ground-truth JSON dataset. When you click evaluate, "
        "the document will be freshly indexed using your custom parameters below, followed by sequential evaluation."
    )
    
    # ── Sidebar: Evaluation Parameters ────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Evaluation Settings")
    st.sidebar.caption("Tune these parameters to see how they impact your RAG metrics.")
    
    # Indexing Parameters
    chunk_size = st.sidebar.slider(
        "Eval Chunk Size", 
        min_value=100, max_value=2000, value=500, step=100,
        help="Characters per chunk. Re-indexes the document on run."
    )
    chunk_overlap = st.sidebar.slider(
        "Eval Chunk Overlap", 
        min_value=0, max_value=500, value=100, step=50,
        help="Character overlap between chunks. Re-indexes the document on run."
    )
    
    st.sidebar.markdown("---")
    
    # Generation/Retrieval Parameters
    top_k = st.sidebar.slider(
        "Eval Top-K Chunks", 
        min_value=1, max_value=20, value=5,
        help="Number of chunks retrieved from ChromaDB per question."
    )
    temperature = st.sidebar.slider(
        "Eval Temperature", 
        min_value=0.0, max_value=1.0, value=0.0, step=0.1,
        help="Keep at 0.0 for strict, deterministic factual testing."
    )

    st.markdown("---")
    
    # ── Uploaders ─────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**1. Source Document**")
        uploaded_doc = st.file_uploader(
            "Upload the text/PDF to be indexed:", 
            type=[ext.lstrip(".") for ext in rg.SUPPORTED_EXTENSIONS],
            key="eval_doc_uploader"
        )
        
    with col2:
        st.markdown("**2. Evaluation Dataset**")
        uploaded_json = st.file_uploader(
            "Upload the JSON Q&A dataset:", 
            type=["json"],
            key="eval_json_uploader"
        )
    
    st.markdown("---")
            
    # ── Execution Trigger ─────────────────────────────────────────────────────
    if st.button("🚀 Index Document & Run Evaluation", type="primary", use_container_width=True):
        
        if not uploaded_doc or not uploaded_json:
            st.error("⚠️ Please upload BOTH the Source Document and the JSON Dataset to begin.")
            return
            
        try:
            eval_data = json.load(uploaded_json)
        except Exception as e:
            st.error(f"Failed to load JSON: {e}")
            return
            
        results = []
        total_q = len(eval_data)
        
        # Setup real-time UI placeholders
        progress_bar = st.progress(0, text="Step 1: Indexing document into ChromaDB...")
        
        st.markdown("### 📈 Live System Averages")
        metrics_placeholder = st.empty()
        
        st.markdown("### 📊 Live Evaluation Results")
        table_placeholder = st.empty()
        
        # Helper for live averages
        def safe_mean(df, col_name):
            if col_name not in df.columns:
                return 0.0
            numeric_vals = pd.to_numeric(df[col_name], errors='coerce')
            return numeric_vals.mean()
        
        # ── Step 1: Re-Index the Document ─────────────────────────────────────
        try:
            eval_collection, n_chunks = rg.build_index(
                uploaded_doc, 
                embed_model=embed_model,
                chunk_size=chunk_size,
                overlap=chunk_overlap
            )
        except Exception as e:
            st.error(f"❌ Indexing failed: {e}")
            progress_bar.empty()
            return
            
        # ── Step 2: Sequential Evaluation Loop ────────────────────────────────
        for i, item in enumerate(eval_data):
            progress_bar.progress((i) / total_q, text=f"Step 2: Processing Question {i+1}/{total_q}...")
            
            question = item.get("question", "")
            ground_truth = item.get("ground_truth_answer", "")
            keywords = item.get("expected_keywords", [])
            
            # 1. Ask the RAG pipeline using the fresh collection and sidebar parameters
            ai_answer, chunks = rg.rag_chat(
                collection=eval_collection,
                user_question=question,
                history=[], 
                chat_model=chat_model,
                embed_model=embed_model,
                top_k=top_k, 
                temperature=temperature 
            )
            
            combined_context = " ".join(chunks)
            
            # 2. Run the math & neural metrics
            retrieval_stats = ev.evaluate_retrieval(chunks, keywords)
            f1 = ev.calculate_token_f1(ai_answer, ground_truth)
            ntjs = ev.non_trivial_jaccard(ai_answer, ground_truth)
            dgi = ev.deterministic_grounding_index(ai_answer, combined_context)
            cos_sim = ev.cosine_similarity(ai_answer, ground_truth, embed_model)
            b_score = ev.calculate_bertscore(ai_answer, ground_truth)
            g_eval = ev.g_eval_score(question, ai_answer, ground_truth, chat_model)
            
            display_b_score = b_score if b_score != -1.0 else "N/A"
            
            # 3. Store row
            results.append({
                "Category": item.get("category", "N/A"),
                "Question": question,
                "Predefined Answer": ground_truth,
                "AI Answer": ai_answer,
                "Retrieval Rank": retrieval_stats["Rank"],
                "Recall@K": retrieval_stats["Recall@K"],
                "MRR": retrieval_stats["MRR"],
                "Token F1": f1,
                "BERTScore": display_b_score,
                "Cosine Sim.": cos_sim,
                "NTJS (Custom)": ntjs,
                "Grounding (DGI)": dgi,
                "G-Eval (1-5)": g_eval
            })
            
            # 4. Update the UI sequentially
            df_results = pd.DataFrame(results)
            
            # Apply styling to force text wrapping
            styled_df = df_results.style.set_properties(**{
                'white-space': 'normal',
                'word-wrap': 'break-word',
                'text-align': 'left'
            })
            
            # Render the updated table with the styles applied
            table_placeholder.dataframe(styled_df, use_container_width=True)
            # Render the updated running averages
            with metrics_placeholder.container():
                cols = st.columns(6)
                cols[0].metric(label="Avg MRR", value=f"{safe_mean(df_results, 'MRR'):.2f}")
                cols[1].metric(label="Avg Token F1", value=f"{safe_mean(df_results, 'Token F1'):.2f}")
                cols[2].metric(label="Avg NTJS", value=f"{safe_mean(df_results, 'NTJS (Custom)'):.2f}")
                cols[3].metric(label="Avg Cosine Sim", value=f"{safe_mean(df_results, 'Cosine Sim.'):.2f}")
                cols[4].metric(label="Avg DGI", value=f"{safe_mean(df_results, 'Grounding (DGI)'):.2f}")
                cols[5].metric(label="Avg G-Eval", value=f"{safe_mean(df_results, 'G-Eval (1-5)'):.2f}/5")
            
        # Finish up
        progress_bar.progress(1.0, text=f"✅ Evaluation Complete! Document was indexed into {n_chunks} chunks.")
        st.balloons()
        
        # Add download button for thesis reporting
        st.markdown("---")
        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name='rag_evaluation_results.csv',
            mime='text/csv',
            use_container_width=True
        )