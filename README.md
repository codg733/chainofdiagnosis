# 🩺 Chain-of-Diagnosis (CoD)

**Chain-of-Diagnosis (CoD)** is an AI-based medical diagnostic system that mimics expert clinical reasoning by combining retrieval, iterative follow-up questioning, and confidence-based reasoning to predict possible diseases based on patient symptoms.

---

## ✨ Key Features & Contributions

- **Unified Synthetic–Real Data Pipeline**  
  Combines synthetic CoD datasets with real-world clinical dialogues (MedDialog) by converting real conversations into structured CoD-style training samples.

- **Demographic-Aware Diagnostic Reasoning**  
  Incorporates patient age and gender as explicit inputs to improve clinical realism and diagnostic relevance.

- **Voice-Enabled Patient Interaction**  
  Supports voice input, allowing users to describe symptoms and details through speech for better accessibility.

- **Hybrid Disease Retrieval System**  
  Uses a weighted combination of BM25 and dense semantic embeddings (FAISS) for robust disease candidate recall.

- **Explainable Multi-Step Reasoning**  
  Produces interpretable reasoning traces, confidence scores, and follow-up questions aligned with the CoD framework.

---

## 🧠 Chain-of-Diagnosis Pipeline

1. Symptom abstraction from text or voice input  
2. Candidate disease retrieval using hybrid search  
3. Diagnostic reasoning and disease scoring  
4. Confidence assessment with probability distribution  
5. Decision making via follow-up questioning or final diagnosis  

---

## 🧱 Models Used

- **Base Model:** Qwen/Qwen2.5-7B-Instruct  
- **Fine-Tuning:** LoRA (PEFT) with Supervised Fine-Tuning (TRL)  
- **Dense Embeddings:** Sentence-Transformers (all-mpnet-base-v2)

---

## 🔍 Retrieval System

- **Lexical Retrieval:** BM25 (rank-bm25)  
- **Dense Retrieval:** all-mpnet-base-v2 embeddings  
- **Vector Search:** FAISS  
- **Hybrid Retrieval:** BM25 + Dense embeddings (weighted)

---

## 📊 Datasets

- **CoD-PatientSymDisease** – Synthetic diagnostic reasoning dataset  
- **MedDialog** – Real-world doctor–patient dialogues  
- **Disease_Database** – Disease–symptom knowledge base  

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **LLM & Training:** Hugging Face Transformers, TRL, PEFT (LoRA)  
- **Retrieval:** BM25, Sentence-Transformers, FAISS  
- **Frontend / UI:** Streamlit  
- **Voice Input:** streamlit_mic_recorder  
- **Utilities:** langdetect, deep_translator  
- **Development:** VS Code, Jupyter, Kaggle  

---

## 🖥️ User Interface

- Interactive Streamlit-based diagnostic application  
- Supports text and voice input  
- Accepts patient age and gender  
- Displays follow-up questions, reasoning, and confidence scores  

---

## 📂 Repository Structure

- [**Dataset**](./Dataset) – Contains all datasets used for training and evaluation  
- [**Resources**](./Resources) – Pre-trained models, embeddings, and retriever cache  
- [**Architecture**](./Architecture) – Retriever, reasoner, and CoD pipeline implementation  
- [**UI**](./ui) – Streamlit application entry point  







