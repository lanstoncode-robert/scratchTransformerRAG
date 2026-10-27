# Custom From-Scratch Transformer RAG System

A lightweight, zero-dependency Retrieval-Augmented Generation (RAG) system built entirely from scratch in PyTorch without using pre-trained weights (No OpenAI, Llama, Qwen, or Hugging Face assets). This repository documents the step-by-step optimization iterations to train a compact deep-learning architecture natively on localized JSON payloads.

---

## 🛠️ Project Architecture & Workflow Map

The framework maps directly to a localized production RAG pattern:

```text
[ Ingestion Layer: JSON Parsing ]
              │
              ▼
[ Word-Level Tokenizer (Vocabulary Index Map) ]
              │
              ▼
[ Jaccard Retrieval Node (Find Context Matches) ]
              │
              ▼
[ Auto-Regressive Sequence Builder (Stitch Prompt + Target) ]
              │
              ▼
[ Causal Transformer Decoder (Compute Next-Token Logits) ]
```


---

## 📋 Daily Log & Engineering Observations

### Environment Configuration
* **Hardware Compute Grid:** NVIDIA GeForce RTX 5050 Laptop GPU (CUDA Acceleration verified).
* **Package Context:** PyTorch Core (v2.11.0+cu128) and `pypdf` (v6.14.2).

### Development Iterations & Core Observations

#### 🔄 Iteration 1
* **Status:** ❌ Did Not Run
* **Observation:** Initial workspace failure due to missing dependencies and script pathways.

#### 🔄 Iteration 2
* **Status:** ⚠️ Executed with Exceptions
* **Observation:** Encountered structural vocabulary translation bugs. The baseline tokenization matrix script hit a catastrophic `KeyError: 'query'` caused by a dictionary field mismatch during text corpus aggregation.

#### 🔄 Iteration 3
* **Status:** ❌ Did Run (Underfitting / Vocabulary Collapse)
* **Generated Response Snippet:** `"the the a a the saha the a the the the the the saha saha..."`
* **Observation:** Upgraded tokenization logic from character-level arrays to a clean **Word-Level Vocabulary Engine** (tracking 226 unique words across 230 matrix input dimensions). While the token pipeline stabilized, the compact network configuration suffered from vocabulary collapse, outputting repetitive functional vocabulary loops (`the`, `a`, `saha`).

#### 🔄 Iteration 4
* **Status:** ❌ Did Run (Hyperparameter Expansion / Structural Misalignment)
* **Generated Response Snippet:** `"the the the the the the the the the the..."`
* **Observation:** Model hidden state layout dimension capacities were intentionally scaled up (`embedding_dim=128`, `hidden_dim=256`, 4 transformer layers deep) to force local memorization profiles. The causal attention mechanisms successfully loaded into CUDA VRAM layers, but highly repeated token distribution behaviors persisted. The root cause was identified as an architectural disconnect: feeding separate input and target arrays prevented the causal network from correlating the query template to the answer path.

#### 🔄 Iteration 5 (Final Stable Phase)
* **Status:**  Did Run Successfully (Unified Auto-Regressive Sequence Mapping)
* **Generated Response Snippet:** `"the cbi arrested two individuals naresh from chennai and sanjib saha from kolkata they used shell companies and mule accounts to launder approximately ₹2 crore"`
* **Observation:** **Success achieved.** Redesigned the PyTorch dataset collection pipeline into a unified auto-regressive tensor sequence, pairing the text context, user query, and target answers into a single index map vector. Implemented clean causal matrix tensor token shifting (`inputs = tokens[:, :-1]` paired against `targets = tokens[:, 1:]`). This architectural synchronization successfully dropped cross-entropy errors and unlocked fluent, grammatically stable target response output validation sequences.

---

## 🔍 Detailed Component & Execution Breakdown

### Step 1: Environment & GPU Check (`Cells 1 & 2`)
The notebook verifies package imports (`torch`, `pypdf`) and queries the system's hardware configuration. When a compatible GPU is detected (`torch.cuda.is_available()`), the model parameters scale across the **NVIDIA GeForce RTX 5050 Laptop GPU** execution grid via CUDA acceleration.

### Step 2: Data Acquisition & Structural Parsing (`Cell 3`)
The system loads local data nodes using standard Python I/O:
* **`knowledge_base.json`**: Acts as the document storage layer containing facts, metadata, and unique location profiles.
* **`evaluation_set.json`**: Houses testing QA vectors, mapping user queries directly to ground truth evaluation answers.

### Step 3: Word-Level Vocabulary Engine & Prompt Patching (`Cell 4`)
The engine iterates over the text corpus to build an isolated index map:
1. Aggregates all words from both JSON configurations into a single raw text sequence.
2. **RAG Prompt Layout Fix:** Manually injects structural tokens (`context`, `query`, `answer`, `|`, `->`) directly into the corpus stream. This guarantees they receive dedicated matrix tracking IDs and prevents vocabulary compilation drops during sequence translation.
3. Lowercases strings, strips trailing punctuation elements (`.,;:!?()"`), isolates unique items, and appends special control markers (`<pad>`, `<sos>`, `<eos>`, `<unk>`).
4. Exposes `encode_text()`, which maps tokenized string vectors into continuous integer ID rows padded uniformly out to a boundary length of 128.

### Step 4: Zero-Dependency Jaccard Retrieval Node (`Cell 5`)
Functions as the framework’s embedded "Vector DB Matcher" without calling external API architectures:
* When given a user question, it breaks the string into a set of distinct lowercase words.
* Loops through the `knowledge_base.json` entries linearly, extracting a target word set for each element.
* Computes mathematical **Jaccard Similarity** ($Score = \frac{|Intersection|}{|Union|}$). 
* Surfaces the absolute maximum overlapping text chunk to serve as context for the downstream decoder.

### Step 5: Combined Auto-Regressive Dataset Pipe (`Cell 6`)
To execute text generation under a causal transformer setup, the prompt and answer are not kept separate. The `RAGDataset` class stitches them into a unified continuous training matrix:
$$\text{Sequence Vector} = [\text{<sos>}] + \text{Context Text} + \text{Query Text} + \text{Answer Label} + \text{Ground Truth Text} + [\text{<eos>}]$$
This ensures the multi-head attention matrix learns exactly how to transition past the prompt structural boundary lines during optimization.

### Step 6: Mini-Causal Decoder Transformer Architecture (`Cell 7`)
Built as a custom PyTorch subclass `nn.Module`, configuring:
* **`nn.Embedding` Layers**: Combines standard token index lookup scores with spatial positional tracking encodings.
* **`nn.TransformerEncoder`**: Structured 4 layers deep using 4 distinct attention heads and a feedforward matrix scale of 256.
* **Causal Masking Layer**: Dynamically evaluates `nn.Transformer.generate_square_subsequent_mask(seq_len)`. This generates a strict lower-triangular mask, blocking tokens during forward training execution loops from viewing future token index allocations.
* **`nn.Linear` Projection Head**: Maps structural transformer outputs back to global vocabulary dimensions.

### Step 7: Optimization Execution Loop via Token Shifting (`Cell 9`)
The dataset is passed to `nn.CrossEntropyLoss(ignore_index=PAD_IDX)` using the **AdamW** optimizer calibrated at a learning rate of $0.0005$. The text sequence is processed using a causal shift alignment:
* **`inputs = tokens[:, :-1]`**: Tokens evaluated from position $0$ to $N-1$.
* **`targets = tokens[:, 1:]`**: Target tokens evaluated from position $1$ to $N$.
By shifting targets one index step forward, the model is trained to predict the **next logical word** across the auto-regressive context line. The error drops down cleanly over 400 epochs, locking weights safely to local context coordinates.

### Step 8: Autoregressive Inference Loop (`Cell 11`)
During inference evaluation:
1. The Jaccard engine pulls the closest document context from storage.
2. The notebook constructs the initial prompt layout up to the `-> Answer: ` block.
3. The forward math pass evaluates the tensor, reading output scores exclusively for the absolute last token position (`logits[0, -1, :]`).
4. **Rigid Argmax Decoding** (`torch.argmax`) isolates the single highest-scoring token ID, appends it dynamically to the input tensor sequence, slides the context window forward, and decodes the next word until an `<eos>` token terminates the generation loop.

---

## 📊 Performance Matrix & Key Takeaways

| Metric Dimension | Core State Value |
| :--- | :--- |
| Unique Word Count | 668 Words |
| Input Tensor Vector Boundaries | 672 Total Dimensions |
| Multi-Head Layout Setup | 4 Attention Heads |
| Dataset Extraction Volume | 19 Knowledge Chunks / 17 Evaluation Queries |
| Final Parameter Error Rate (Loss) | ~0.0338 (Highly Converged) |
| Latency Target Compute Engine | `cuda` (NVIDIA GeForce RTX 5050 Laptop GPU) |

* **Key Takeaway:** Building text generators completely from scratch means the model begins with absolute blank semantic awareness. Iteration 5 proved that causal decoders require unified auto-regressive contexts combined with strict input-target sequence shifting to correctly link retrieved facts to grammatically sound answer strings.