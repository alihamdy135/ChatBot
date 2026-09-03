# RAG E-commerce Support Chatbot — Fully Local Workflow

Everything runs on your own machine: **training and inference**. Datasets are
pulled via the HuggingFace `datasets` API (no manual downloads). The only thing
NOT local is the final answer-generation call, which uses the Groq cloud API —
this keeps your 4GB GPU free for the training steps and avoids fitting an LLM
into a very tight VRAM budget.

Your hardware: GPU 4GB VRAM, 32GB RAM, 9-core CPU — notebooks below are tuned
for this.

```
training_notebooks/   <- run once, locally, to produce model artifacts
local_app/              <- the FastAPI chatbot that uses those artifacts
```

---

## STEP 0 — One-time environment setup

```bash
cd local_app
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install PyTorch WITH CUDA support first (see requirements.txt comment for the
# exact command — check your CUDA version with `nvidia-smi` first)
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
pip install jupyter jupytext   # to run the training notebooks
```

Verify GPU is visible to PyTorch:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
If this prints `False`, the DistilBERT notebook will still work — just slower
(CPU fallback is built in).

## STEP 1 — Train all four models locally

Open Jupyter from the project root:
```bash
jupyter notebook training_notebooks/
```

Run each notebook top to bottom, in this order:

| Notebook | Uses GPU? | Approx. time on your hardware | Saves to |
|---|---|---|---|
| `01_language_detection.ipynb` | No (CPU, 9 cores) | ~2-5 min | `local_app/models/language/` |
| `03_intent_classifier.ipynb` | No (CPU) | ~3-8 min | `local_app/models/intent/` |
| `04_rag_build_index.ipynb` | Optional (embeds faster with GPU) | ~3-10 min | `local_app/models/rag/` |
| `02_sentiment_distilbert.ipynb` | **Yes**, tuned for 4GB VRAM | ~20-45 min | `local_app/models/sentiment/` |

Each notebook writes its outputs **directly** into the correct `local_app/models/*`
subfolder via a relative path (`../local_app/models/...`) — no manual file-moving
needed, unlike a Kaggle workflow.

### If notebook 2 (sentiment) hits `CUDA out of memory`
The notebook already uses small-batch + gradient accumulation + fp16 settings
built for 4GB cards. If you still OOM:
1. Close other GPU-using apps (browser hardware acceleration, games, etc.)
2. In the notebook, drop `per_device_train_batch_size` from 8 to 4 and raise
   `gradient_accumulation_steps` from 4 to 8 (keeps the same effective batch size)
3. As a last resort, force CPU by setting `use_fp16 = False` and moving the model
   to CPU manually — much slower but will not OOM

## STEP 2 — Get a Groq API key (only cloud dependency, used for final answer generation)

1. Sign up free at [console.groq.com](https://console.groq.com)
2. Create an API key
3. **If you ever paste a key into a chat, an email, or commit it to git, treat it
   as compromised and regenerate it immediately** — keys should only ever live
   in your local `.env` file.

## STEP 3 — Configure and run the app

```bash
cd local_app
cp .env.example .env
# edit .env, paste your Groq key into GROQ_API_KEY

uvicorn app.main:app --reload --port 8000
```

You should see `"All models loaded. Ready."` — all four models (language,
sentiment, intent, RAG/FAISS) load from the files the notebooks just produced.

## STEP 4 — Test it

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "my order still hasnt arrived and im really annoyed about it"}'
```

The response includes the final answer plus every intermediate signal
(detected language, sentiment, intent, routing decision, retrieved chunks) —
useful for connecting a frontend and for explaining any step in your assessment.

## STEP 5 — Connect your Next.js frontend
`POST http://localhost:8000/chat` with body `{"message": "..."}`. CORS is open
(`allow_origins=["*"]`) for local dev.

---

## Project structure

```
ecommerce-chatbot-local/
├── training_notebooks/
│   ├── 01_language_detection.ipynb
│   ├── 02_sentiment_distilbert.ipynb    <- only one needing GPU
│   ├── 03_intent_classifier.ipynb
│   └── 04_rag_build_index.ipynb
├── local_app/
│   ├── app/
│   │   ├── main.py         <- FastAPI entrypoint, /chat endpoint
│   │   ├── language.py     <- language detection inference
│   │   ├── sentiment.py    <- DistilBERT sentiment inference
│   │   ├── intent.py       <- intent classifier inference
│   │   ├── rag.py          <- FAISS retrieval (local) + Groq generation (cloud)
│   │   └── router.py       <- routing logic, see docstring for design rationale
│   ├── models/               <- populated by the training notebooks, not tracked in git
│   ├── requirements.txt
│   ├── .env.example
│   └── .env                  <- you create this, never commit it
└── README.md
```

## What's local vs. cloud, and why

| Component | Where it runs | Why |
|---|---|---|
| Dataset loading | Local (via HF `datasets` API, cached after first pull) | No manual downloads, but no training data leaves your machine either — the API just fetches once |
| Language detection training + inference | 100% local, CPU | Lightweight TF-IDF model |
| Sentiment training + inference | 100% local, your GPU | DistilBERT fine-tune, tuned for 4GB VRAM |
| Intent training + inference | 100% local, CPU | Lightweight TF-IDF+SVM model |
| Embeddings (MiniLM) + FAISS retrieval | 100% local, GPU or CPU | Small model, exact search |
| Final answer generation | **Cloud (Groq API)** | Only component that needs a large LLM; running an LLM good enough for grounded generation locally on 4GB VRAM would require heavy quantization and produce noticeably worse answers than a hosted model |

If you later want generation local too (e.g. via Ollama with a small quantized
model), that's a drop-in swap in `app/rag.py` — say the word and I'll build that
version as an alternative.

## Key design decisions (for your assessment)
Same as before — each notebook's markdown cells document its "WHY". Summary:
- **Language detection**: TF-IDF char n-grams + Logistic Regression.
- **Sentiment**: fine-tuned DistilBERT, 6 emotions → 3 buckets, VRAM-safe training
  settings (small batch + gradient accumulation + fp16), domain-shift (Twitter→
  support) documented and sanity-checked in the notebook.
- **Intent**: TF-IDF + Linear SVM on all 27 fine intents, mapped to 7 routing
  buckets at inference.
- **RAG**: FAISS flat index, all-MiniLM-L6-v2 embeddings (local), Groq gpt-oss-120b
  generation (cloud, only cloud component).
- **Complaint/negative-sentiment routing**: apology-prefixed RAG answer + priority
  flag rather than blocking auto-response — rationale in `app/router.py`.
