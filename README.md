# PolarityIQ · Family Office Intelligence RAG Pipeline

> **Task #2 — RAG Pipeline Build**
> 200 real family offices. Queryable in natural language. Advanced RAG techniques.

---

## Live Demo

**Streamlit App:** _(deploy link appears here after deployment)_

---

## Project Structure

```
PolarityIQ_RAG/
├── app.py                          # Streamlit UI (professional interface)
├── rag_pipeline.py                 # Core RAG engine (all advanced techniques)
├── requirements.txt                # Python dependencies
├── .gitignore
├── .streamlit/
│   ├── config.toml                 # Theme + server settings
│   └── secrets.toml.example        # API key template (copy → secrets.toml)
├── data/
│   └── PolarityIQ_FamilyOffice_200_CLEAN_v3.xlsx   # Task #1 dataset
├── polarityiq_chromadb/            # Pre-built ChromaDB vector store (200 records)
└── notebooks/
    └── polarityiq_rag_pipeline_FINAL.ipynb          # Full experiment notebook
```

---

## Quick Start (Local)

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/polarityiq-rag.git
cd polarityiq-rag

# 2. Create virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set API key
export ANTHROPIC_API_KEY="sk-ant-..."

# 5. Run the app
streamlit run app.py
```

The ChromaDB vector store is pre-built and included in the repo.
If it's missing, the pipeline auto-rebuilds it from the Excel file in ~1 second.

---

## Streamlit Cloud Deployment

1. Push this repo to GitHub (public or private)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select repo → branch `main` → main file: `app.py`
4. In **Advanced settings → Secrets**, paste:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
5. Click **Deploy**

---

## Stack Choices & Justification

| Component | Choice | Why |
|---|---|---|
| Vector DB | ChromaDB (persistent) | Zero infra, local file, metadata filtering, cosine similarity |
| Embedding model | BAAI/bge-small-en-v1.5 | Highest MTEB retrieval score in its size class (60.1 avg), 2.5× faster than mpnet |
| Reranker | ms-marco-MiniLM-L-6-v2 | Lightweight cross-encoder, trained on 500k MS MARCO passage pairs |
| Lexical retrieval | BM25Okapi (rank-bm25) | Handles exact name/entity matching that vector search misses |
| LLM | claude-sonnet-4-6 | Best instruction following + grounding enforcement via Anthropic API |
| CRAG grader | claude-haiku-4-5-20251001 | 20× cheaper than Sonnet, accurate enough for binary relevance classification |
| Framework | Pure Python (no LangChain) | Fully debuggable, minimal abstraction leakage, easier to explain and extend |

---

## Advanced RAG Techniques Implemented

### 1. Hybrid Retrieval (Vector + BM25)

**Problem:** Pure vector search misses exact name matches. Pure BM25 misses semantic similarity.

**Solution:** ChromaDB cosine vector search retrieves 20 candidates. BM25 (with domain-specific
stopword removal) re-scores them. Final score:

```
score = 0.7 × vector_score + 0.3 × min(bm25_score, 0.6)
```

The BM25 cap (0.6) prevents keyword-flooded records from dominating.
Domain stopwords (`family`, `office`, `capital`, etc.) are removed — they appear in
every record and would otherwise inflate BM25 scores for unrelated results.

**Result:** 6/6 = **100% hit rate** on ground-truth evaluation queries.

---

### 2. Cross-Encoder Reranking

**Paper:** Standard information retrieval technique, popularised by Nogueira & Cho (2019).

**Problem:** Bi-encoders (like BGE) encode query and document **separately** and compare
with dot product — a shallow relevance signal.

**Solution:** Cross-encoder feeds **(query, document)** together to a transformer that
attends over both simultaneously — much richer joint relevance scoring.

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M parameters)

**Result:** Top-5 rankings changed substantially in testing — one document jumped from rank #7
to rank #1 after reranking, changing the final answer.

**Flow:** Hybrid retrieval fetches 15 candidates → cross-encoder reranks → top 5 returned.

---

### 3. CRAG — Corrective RAG

**Paper:** "Corrective Retrieval Augmented Generation" (Yan et al., 2024)
[arXiv:2401.15884](https://arxiv.org/abs/2401.15884)

**Problem:** Standard RAG blindly trusts whatever the retriever returns. If retrieval
fails, the LLM hallucinates using off-topic context.

**Solution:** After retrieval, Claude Haiku grades each document:
- `relevant` → keep
- `partially_relevant` → keep
- `irrelevant` → filter out

If ALL documents are irrelevant → return an honest "not found" response instead of
generating a hallucinated answer.

**Why Haiku for grading?** It's 20× cheaper than Sonnet and relevance classification
is a simple task — binary signal, not synthesis.

---

### 4. RAG Fusion + Reciprocal Rank Fusion (RRF)

**Papers:**
- "RAG-Fusion" (Shi et al., 2024)
- "Reciprocal Rank Fusion" (Cormack, Clarke, Buettcher, 2009)

**Problem:** A single query has one perspective. "AI tech family offices" misses
documents that say "artificial intelligence" or "machine learning" investments.

**Solution:**
1. Claude Haiku generates 3 alternative phrasings of the query
2. Retrieve independently for each variant (4 queries total)
3. Fuse ranked lists using RRF:

```
score(d) = Σ  1 / (k + rank(d))    k=60 (empirically validated constant)
```

Documents appearing in multiple ranked lists get boosted — surfacing records
that no single query would have found alone.

4. Cross-encoder reranks the fused pool.

---

### 5. SQL Router — Exact Numeric Filtering

**Problem:** Vector search has no concept of numeric ordering.
"Check size > $10M" may retrieve a $8M record and $15M record with equal probability.

**Solution:** Classify query type:
- `numeric` — has threshold/range keywords → **pandas `.query()` filter** (exact)
- `semantic` — pure meaning query → **RAG Fusion** (semantic)
- `hybrid` — both → run both paths, merge results

Claude Haiku generates a pandas `.query()` expression. A security validator
strips quoted string values before scanning for unknown column identifiers —
preventing column injection from LLM-generated code.

**Example:**
```
Query:  "US family offices with minimum check above $10M"
Filter: Country == 'USA' and Check_Size_Min_M >= 10
Result: exact match, 100% precise
```

---

### 6. Self-RAG — Confidence Scoring

**Paper:** "Self-RAG: Learning to Retrieve, Generate, and Critique through
Self-Reflection" (Asai et al., ICLR 2024) [arXiv:2310.11511](https://arxiv.org/abs/2310.11511)

**What our implementation evaluates (post-generation):**

| Signal | Type | Description |
|---|---|---|
| `confidence_score` | 1–5 integer | How well the context supports the answer |
| `fully_grounded` | boolean | Are all claims traceable to retrieved documents? |
| `coverage_gaps` | string | Query aspects the dataset couldn't answer |
| `advice` | string | How to improve the response quality |

**Why this matters:** A system that knows what it *doesn't* know is more valuable
than one that confidently gives wrong answers. Confidence scores let users decide
whether to trust the output or verify further.

---

## Chunking Strategy

**Strategy C: Structured Narrative Paragraph** (winner from 3-strategy experiment)

Each family office record becomes one dense natural-language paragraph (~267 tokens):

```
Walton Enterprises LLC [PIQ-001] is a SFO family office headquartered in
Bentonville, USA, founded in 1953. Wealth creator: Sam Walton. Wealth source:
Retail / Walmart. Estimated AUM: $100B+ (high confidence). Primary strategy:
Diversified Long-Term. Sector focus: Retail, Real Estate, Media...
```

**Why not Strategy A (key=value flat dump)?**
Machine-readable but poor embedding quality — LLMs generate better answers
when context reads like prose.

**Why not Strategy B (4 sub-chunks per record)?**
Cross-field relationships (e.g. "AI sector + check size > $10M") end up in
different vectors, making joint queries harder.

**Why 1 chunk per record instead of splitting?**
At ~267 tokens, each record fits comfortably within the attention window.
Splitting would break the narrative and require chunk-level retrieval
that loses the holistic record view.

---

## Evaluation Results

### Retrieval Hit Rate (Baseline)

| Query | Expected | Hit (Top-5) |
|---|---|---|
| Andreessen Horowitz Silicon Valley AI crypto | PIQ-147 | ✅ |
| Crypto bitcoin blockchain Gemini exchange | PIQ-089 | ✅ |
| South African mining diamonds De Beers | PIQ-131 | ✅ |
| Brazilian steel conglomerate industrial | PIQ-143 | ✅ |
| UAE royal family office Abu Dhabi tech | PIQ-058 | ✅ |
| Walton family Walmart heirs Bentonville | PIQ-001 | ✅ |

**Baseline hit rate: 6/6 = 100%**

### Cross-Encoder Impact (example query)

Query: *"European family office focused on climate tech and ESG impact investing"*

| Rank | Baseline (hybrid) | After Cross-Encoder |
|---|---|---|
| #1 | PIQ-039 (Exor NV) | PIQ-007 (Hillspire LLC) ↑ from #7 |
| #2 | PIQ-055 (Brenninkmeyer) | PIQ-054 (Wirtgen Invest) |
| #3 | PIQ-054 (Wirtgen) | PIQ-039 (Exor NV) ↓ |

Hillspire moved 6 positions after the cross-encoder read query and document jointly.

---

## What Failed / Limitations

1. **Email completeness** — Most SFOs don't publish contact information publicly.
   DM1_Email fill rate is ~40%. Production enrichment would use Apollo.io or Hunter.

2. **AUM is estimated, not audited** — Family offices are private. AUM figures come
   from news reports, interviews, and public filings. Ranges are used, not exact figures.

3. **CRAG grader on API errors** — When Haiku is unavailable (billing, rate limits),
   the grader defaults to "relevant" for all documents. CRAG loses its filtering power
   but the pipeline continues running.

4. **Query expansion on API errors** — RAG Fusion falls back to single-query mode
   (no expansion). Still works, just without multi-perspective retrieval.

5. **No cross-record aggregation** — COUNT/SUM/GROUP BY queries are not supported.
   A SQL layer (PostgreSQL + pgvector) would enable analytics-style queries.

---

## What I Would Improve with More Time

1. **pgvector + PostgreSQL** — Replace ChromaDB with PostgreSQL + pgvector. Enables
   exact numeric filters in SQL *combined* with vector similarity in one query.
   Eliminates the need for the separate SQL router.

2. **Contextual chunk compression** — Before embedding, use an LLM to compress each
   chunk to the most retrieval-relevant sentences. Reduces noise in the embedding.

3. **Adaptive CRAG** — Track grading accuracy over time. Tune the relevance threshold
   per query type based on feedback.

4. **Streaming responses** — Stream Claude's generation token-by-token so the user
   sees the answer build in real time (Streamlit supports `st.write_stream`).

5. **Real-time enrichment** — On retrieval, trigger an Apollo/LinkedIn API call
   to fetch live contact data before displaying results.

6. **Fine-tuned embeddings** — Fine-tune BGE on family office domain pairs
   (query, relevant_record) to improve retrieval for FO-specific terminology.

---

## Contact

Submitted for: PolarityIQ / Falcon Scaling optimization task evaluation.
