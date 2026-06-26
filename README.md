# Custom From-Scratch Transformer RAG System

A lightweight, zero-dependency Retrieval-Augmented Generation (RAG) system built entirely from scratch in PyTorch without using pre-trained weights (No OpenAI, Llama, Qwen, or Hugging Face assets). This repository documents the step-by-step optimization iterations to train a compact deep-learning architecture natively on localized JSON payloads.

---

## 🛠️ Project Architecture & Workflow

1. **Ingestion Layer:** Parses custom structural components directly from raw source files (`knowledge_base.json` and `evaluation_set.json`).
2. **Tokenizer Engine:** Built-from-scratch Word-Level Vocabulary Engine that sanitizes punctuation, handles out-of-vocabulary (`<unk>`) edge cases, and maps texts into sequential tensor indices.
3. **Retrieval Node:** A zero-dependency Jaccard Text Overlap Search Engine used to slice and surface contextual facts based on absolute word intersections.
4. **Mini-Causal Decoder:** A pure PyTorch Causal Language Transformer Model mapping structural spatial multi-head self-attention paths over custom causal lower-triangular mask matrices.

---

## 📋 Daily Log & Engineering Observations

### Environment Configuration
* **Hardware Compute Grid:** NVIDIA GeForce RTX 5050 Laptop GPU (CUDA Acceleration verified).
* **Package Context:** PyTorch Core (v2.12.1+cu126) and `pypdf` installed within localized Python virtual environment (`venv1`).

### Development Iterations & Core Observations

#### 🔄 Iteration 1
* **Status:** ❌ Did Not Run
* **Observation:** Initial workspace failure due to missing dependencies and script pathways.

#### 🔄 Iteration 2
* **Status:** ⚠️ Executed with Exceptions
* **Observation:** Encountered structural vocabulary translation bugs. The baseline tokenization matrix script hit a catastrophic `KeyError: 'query'` caused by a dictionary field mismatch during text corpus aggregation.

#### 🔄 Iteration 3
* **Status:**  Did Run (Underfitting / Vocabulary Collapse)
* **Generated Response Snippet:** `"the the a a the saha the a the the the the the saha saha..."`
* **Observation:** Upgraded tokenization logic from character-level arrays to a clean **Word-Level Vocabulary Engine** (tracking 226 unique words across 230 matrix input dimensions). While the token pipeline stabilized, the compact network configuration suffered from vocabulary collapse, outputting repetitive functional vocabulary loops (`the`, `a`, `saha`).

#### 🔄 Iteration 4
* **Status:**  Did Run (Hyperparameter Expansion)
* **Generated Response Snippet:** `"the the the the the the the the the the..."`
* **Observation:** Model hidden state layout dimension capacities were intentionally scaled up (`embedding_dim=128`, `hidden_dim=256`, 4 transformer layers deep) to force local memorization profiles. The causal attention mechanisms successfully loaded into CUDA VRAM layers, but highly repeated token distribution behaviors indicate a missing temperature scaling layer or an unbalanced cross-entropy token penalty parameter during the backward step.

---

## 🚀 Step-by-Step Execution Guide

### 1. Ingestion Requirements
Ensure your workspace directory contains your source dictionary layers formatted precisely like this:

**`knowledge_base.json`**
```json
[
    {
        "chunk_id": 1,
        "text": "As part of Operation Chakra-VI, the CBI arrested two key accused: Naresh from Chennai and Sanjib Saha..."
    }
]
```

### 2. Environment Verification
Open your VS Code terminal and verify package bindings using the diagnostic cell configuration:
```python
try:
    import torch
    import pypdf
    print(f"✅ Success! torch version: {torch.__version__} | GPU Available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"❌ Error: {e}")
```

### 3. Native Model Execution Loop
To re-run text corpus tensor collection, local dataset token mapping, Jaccard contextual scoring, and local network training, invoke the pipeline execution via:
```bash
python train_scratch_transformer_rag.py
```

---

## 📊 Performance Matrix & Key Takeaways

| Metric Dimension | Core State Value |
| :--- | :--- |
| Unique Word Count | 226 Words |
| Input Tensor Vector Boundaries | 230 Total Dimensions |
| Multi-Head Layout Setup | 4 Attention Heads |
| Dataset Extraction Volume | 6 Knowledge Chunks / 5 Evaluation Queries |
| Latency Target Compute Engine | `cuda` (NVIDIA GeForce RTX 5050) |

* **Key Takeaway:** Building text generators completely from scratch means the model begins with absolute blank semantic awareness—it must discover basic structural English syntax solely from your JSON arrays. Because the local matrix profile is compact, minor variances in loss convergence easily tilt generation layers between token collapse modes and raw overfitted memorization.
"# scratchTransformerRAG" 
