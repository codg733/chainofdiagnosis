# backend/model_loader.py

import os, pickle, re
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ---------------------------------------
# CONFIG
# ---------------------------------------
try:
    import config
except ImportError:
    # Fallback if running as script from backend folder
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config

BASE_MODEL = config.BASE_MODEL_NAME
MODEL_DIR = config.MODEL_ADAPTER_DIR

RETRIEVER_CACHE = config.RETRIEVER_CACHE_DIR
os.makedirs(RETRIEVER_CACHE, exist_ok=True)

# global cached objects
_retriever_loaded = False
kb_df = None
bm25 = None
corpus = None
faiss_index = None
embedder = None
disease_symptom_map = None
# --- ENSURE DISEASE MAP AVAILABLE ---
def ensure_disease_map():
    """
    Safe getter for disease_symptom_map.
    Ensures the retriever is loaded and returns the map.
    """
    from backend.model_loader import load_retriever
    global disease_symptom_map

    if disease_symptom_map is None:
        _, disease_symptom_map, _, _, _, _ = load_retriever()

    return disease_symptom_map


# ---------------------------------------
# LOAD / BUILD RETRIEVER
# ---------------------------------------
def load_retriever():
    """
    Loads cached retriever OR builds one from HF dataset.
    Returns: kb_df, disease_map, bm25, tokenized_corpus, faiss_index, embedder
    """
    global _retriever_loaded, kb_df, bm25, corpus, faiss_index, embedder, disease_symptom_map

    if _retriever_loaded:
        return kb_df, disease_symptom_map, bm25, corpus, faiss_index, embedder

    KB = os.path.join(RETRIEVER_CACHE, "kb.pkl")
    BM25 = os.path.join(RETRIEVER_CACHE, "bm25.pkl")
    CORPUS = os.path.join(RETRIEVER_CACHE, "corpus.pkl")
    FAISS = os.path.join(RETRIEVER_CACHE, "faiss.index")
    MAP = os.path.join(RETRIEVER_CACHE, "symptom_map.pkl")

    # -------------- load cache ----------------
    if all(os.path.exists(x) for x in [KB, BM25, CORPUS, FAISS, MAP]):
        kb_df = pd.read_pickle(KB)

        with open(MAP, "rb") as f:
            disease_symptom_map = pickle.load(f)
        with open(BM25, "rb") as f:
            bm25 = pickle.load(f)
        with open(CORPUS, "rb") as f:
            corpus = pickle.load(f)

        faiss_index = faiss.read_index(FAISS)
        embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

        _retriever_loaded = True
        return kb_df, disease_symptom_map, bm25, corpus, faiss_index, embedder

    # -------------- build retriever ----------------
    ds = load_dataset("FreedomIntelligence/Disease_Database", "en", split="train")
    df = pd.DataFrame(ds)

    # detect symptom column
    sym_cols = [c for c in df.columns if "symptom" in c.lower()]
    sym_col = sym_cols[0]
    df["symptom_text"] = df[sym_col].astype(str)

    disease_symptom_map = {row["disease"]: row["symptom_text"] for _, row in df.iterrows()}

    tokenized = [re.findall(r"\w+", s.lower()) for s in df["symptom_text"]]
    bm25 = BM25Okapi(tokenized)

    embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    embs = embedder.encode(df["symptom_text"].tolist(), convert_to_numpy=True)
    faiss.normalize_L2(embs)
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)

    # cache all components
    df.to_pickle(KB)
    with open(MAP, "wb") as f: pickle.dump(disease_symptom_map, f)
    with open(BM25, "wb") as f: pickle.dump(bm25, f)
    with open(CORPUS, "wb") as f: pickle.dump(tokenized, f)
    faiss.write_index(idx, FAISS)

    kb_df = df
    corpus = tokenized
    faiss_index = idx

    _retriever_loaded = True
    return kb_df, disease_symptom_map, bm25, corpus, faiss_index, embedder


# ---------------------------------------
# Ensure disease_symptom_map available
# ---------------------------------------
def ensure_disease_map():
    """Safely loads disease_symptom_map and returns it."""
    global disease_symptom_map
    if disease_symptom_map is None:
        _, disease_symptom_map, _, _, _, _ = load_retriever()
    return disease_symptom_map


# ---------------------------------------
# Hybrid Retrieve
# ---------------------------------------
def hybrid_retrieve(symptoms, k=6, alpha=0.6):
    """
    BM25 + FAISS hybrid retriever with robust preprocessing.
    """
    global kb_df, disease_symptom_map, bm25, corpus, faiss_index, embedder

    load_retriever()  # ensure loaded

    if isinstance(symptoms, list):
        symptoms = ", ".join(symptoms)

    if not symptoms:
        symptoms = "symptoms"

    q = symptoms.lower()
    toks = re.findall(r"\w+", q)

    # BM25
    bm = np.array(bm25.get_scores(toks))
    if bm.max() > 0:
        bm /= (bm.max() + 1e-12)

    # FAISS
    emb = embedder.encode([q], convert_to_numpy=True)
    faiss.normalize_L2(emb)
    sims, ids = faiss_index.search(emb, len(kb_df))
    fs = np.zeros(len(kb_df))
    fs[ids[0]] = sims[0]

    if fs.max() > 0:
        fs /= (fs.max() + 1e-12)

    # Hybrid
    final = alpha * fs + (1 - alpha) * bm
    top_ids = np.argsort(final)[::-1][:k]

    return [kb_df.iloc[i]["disease"] for i in top_ids]


# ---------------------------------------
# (Optional) Load LORA model
# ---------------------------------------
def load_lora_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", trust_remote_code=True)

        model = PeftModel.from_pretrained(base, MODEL_DIR, device_map="auto")
        return model, tokenizer
    except Exception as e:
        print("⚠ LoRA model load failed:", e)
        return None, None
