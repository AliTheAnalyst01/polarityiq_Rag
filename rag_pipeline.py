"""
PolarityIQ RAG Pipeline  ·  rag_pipeline.py
============================================================
Advanced Retrieval-Augmented Generation engine for the
PolarityIQ Family Office Intelligence dataset.

Techniques implemented
─────────────────────
  1. Hybrid Retrieval      — ChromaDB vector + BM25 (domain-stopword tokenised)
  2. Cross-Encoder Rerank  — ms-marco-MiniLM-L-6-v2 joint (query, doc) scoring
  3. CRAG                  — Corrective RAG: relevance grading + fallback
  4. RAG Fusion + RRF      — Multi-query expansion + Reciprocal Rank Fusion
  5. SQL Router            — Exact pandas filter path for numeric/categorical queries
  6. Self-RAG              — Post-generation confidence scoring

Usage
─────
  from rag_pipeline import PolarityIQPipeline
  pipe = PolarityIQPipeline(api_key="sk-ant-...")
  result = pipe.advanced_ask("AI family offices with check > $10M")
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import anthropic as _anthropic_sdk
import chromadb
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

# ── Paths (relative to this file — works locally AND on Streamlit Cloud) ──────
_BASE = Path(__file__).parent
DATA_PATH  = _BASE / "data" / "PolarityIQ_FamilyOffice_200_CLEAN_v3.xlsx"
DB_PATH    = str(_BASE / "polarityiq_chromadb")
COLLECTION = "family_offices_v1"

# ── Model identifiers ──────────────────────────────────────────────────────────
EMBED_MODEL_NAME    = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL           = "claude-sonnet-4-6"
HAIKU_MODEL         = "claude-haiku-4-5-20251001"

# ── BM25 domain stopwords ──────────────────────────────────────────────────────
# Words that appear in almost every FO record — remove from BM25 index so
# they don't inflate similarity scores for unrelated records.
_FINANCE_STOPWORDS = {
    "family", "office", "capital", "group", "fund", "management",
    "holdings", "holding", "investment", "investments", "partners",
    "enterprises", "ventures", "trust", "foundation", "llc", "inc",
    "corp", "corporation", "limited", "ltd", "co", "sfo", "mfo",
    "wealth", "private", "asset", "assets", "portfolio", "invest",
    "investing", "investor", "piq", "n/a", "yes", "no", "both",
    "direct", "verified", "partial", "a", "an", "the", "is", "are",
    "was", "of", "in", "at", "to", "and", "or", "for", "on", "with",
    "by", "from", "its", "it", "this", "that", "as", "be", "has",
    "have", "had", "not", "based", "founded", "approximately",
    "estimated", "typical", "focus", "include", "includes", "including",
}

# ── SQL safe column allowlists ─────────────────────────────────────────────────
_SAFE_NUMERIC = {"Check_Size_Min_M", "Check_Size_Max_M", "Founding_Year"}
_SAFE_STRING  = {
    "Country", "FO_Type", "Sector_Focus", "Geo_Preference",
    "Primary_Strategy", "ESG_Focus", "Co_Invest", "AUM_Range",
    "Asset_Classes", "Wealth_Source_Industry",
}
_ALL_SAFE_COLS  = _SAFE_NUMERIC | _SAFE_STRING
_ALLOWED_TOKENS = {"and", "or", "not", "in", "True", "False", "nan", "null"}

# Patterns that signal a numeric/filter query → SQL routing path
_NUMERIC_PATTERNS = [
    "greater than", "less than", "more than", "above", "below",
    "over", "under", "at least", "at most", "minimum", "maximum",
    ">", "<", ">=", "<=", "between", "range",
    "check size", "aum", "assets under management",
    "billion", "million", "$", "£", "€",
    "founded before", "founded after",
]

# ── LLM system prompt ──────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are the PolarityIQ Intelligence Assistant — an expert analyst for family
offices and private markets.

Answer queries using ONLY the retrieved context documents provided.
NEVER invent or hallucinate facts. If the dataset lacks information, say so.

Format every answer as:
1. **Direct Answer** — 1–2 concise sentences
2. **Matching Records** — markdown table:
   | Record ID | FO Name | Country | AUM Range | Key Detail |
3. **Analyst Commentary** — 2–3 sentences: patterns, caveats, next steps

Always cite Record IDs (PIQ-XXX) for every claim. Use clean markdown."""


# ══════════════════════════════════════════════════════════════════════════════
#  Helper: JSON response parser
# ══════════════════════════════════════════════════════════════════════════════

def _parse_json(text: str) -> dict | list:
    """
    Parse a JSON response that may be wrapped in markdown code fences.

    Claude Haiku sometimes returns:
        ```json
        {"grade": "relevant"}
        ```
    instead of bare JSON. This helper strips the fence before parsing.
    """
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.lstrip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rstrip("`").strip()
    return json.loads(raw)


# ══════════════════════════════════════════════════════════════════════════════
#  Chunking  (Strategy C — Structured Narrative, proven best in experiments)
# ══════════════════════════════════════════════════════════════════════════════

def _build_chunk(row: dict) -> str:
    """Convert one DataFrame row into a dense natural-language narrative chunk."""
    def v(field: str, default: str = "N/A") -> str:
        val = str(row.get(field, "")).strip()
        return val if val and val not in ("nan", "None", "") else default

    return (
        f"{v('FO_Name')} [{v('Record_ID')}] is a {v('FO_Type')} family office "
        f"headquartered in {v('City')}, {v('Country')}, founded {v('Founding_Year')}. "
        f"Wealth creator: {v('Wealth_Creator')}. Wealth source: {v('Wealth_Source_Industry')}. "
        f"Estimated AUM: {v('AUM_Range')} (confidence: {v('AUM_Confidence')}). "
        f"Primary strategy: {v('Primary_Strategy')}. "
        f"Asset classes: {v('Asset_Classes')}. "
        f"Sector focus: {v('Sector_Focus')}. "
        f"Geographic preference: {v('Geo_Preference')}. "
        f"Direct or fund: {v('Direct_or_Fund')}. "
        f"Check size: ${v('Check_Size_Min_M')}M – ${v('Check_Size_Max_M')}M. "
        f"Co-investment appetite: {v('Co_Invest')}. "
        f"ESG focus: {v('ESG_Focus')}. "
        f"LP relationships: {v('LP_Relationships')}. "
        f"Sample portfolio: {v('Sample_Portfolio')}. "
        f"Recent signals (2024-2025): {v('Signal_2024_2025')}. "
        f"Decision maker: {v('DM1_Name')} ({v('DM1_Title')}). "
        f"Email: {v('DM1_Email')}. LinkedIn: {v('DM1_LinkedIn')}. "
        f"Website: {v('Website')}. Validation: {v('Validation')}."
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PolarityIQPipeline — main class
# ══════════════════════════════════════════════════════════════════════════════

class PolarityIQPipeline:
    """
    Full advanced RAG pipeline for PolarityIQ family office intelligence.

    Instantiation loads all models and the vector database once.
    All query methods are then fast (no reload on every call).

    Args:
        api_key : Anthropic API key (sk-ant-...)
    """

    def __init__(self, api_key: str) -> None:
        self._client = _anthropic_sdk.Anthropic(api_key=api_key)
        self._load_data()
        self._load_models()
        self._setup_chromadb()
        self._build_bm25()

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_data(self) -> None:
        """Load and clean the Excel dataset; build record dicts + chunk texts."""
        df_raw = pd.read_excel(DATA_PATH, sheet_name="Family Office Intelligence", header=3)
        df_raw.columns = [
            str(c).strip().replace(" ", "_").replace("/", "_").replace("-", "_")
            for c in df_raw.columns
        ]
        df = df_raw[df_raw["FO_Name"].astype(str).str.strip().ne("")].copy()
        df = df.reset_index(drop=True)

        # Numeric coercion for SQL path
        for col in _SAFE_NUMERIC:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Text coercion
        text_cols = df.select_dtypes(include="object").columns
        df[text_cols] = df[text_cols].fillna("")

        self.df = df
        self.records: list[dict] = []
        for i, row in df.iterrows():
            rec = row.to_dict()
            rec_id = str(rec.get("Record_ID", f"PIQ-{i+1:03d}")).strip()
            self.records.append({
                "id":         rec_id,
                "chunk":      _build_chunk(rec),
                "fo_name":    str(rec.get("FO_Name", "")).strip(),
                "fo_type":    str(rec.get("FO_Type", "")).strip(),
                "country":    str(rec.get("Country", "")).strip(),
                "city":       str(rec.get("City", "")).strip(),
                "aum_range":  str(rec.get("AUM_Range", "")).strip(),
                "sector":     str(rec.get("Sector_Focus", "")).strip(),
                "co_invest":  str(rec.get("Co_Invest", "")).strip(),
                "esg":        str(rec.get("ESG_Focus", "")).strip(),
                "strategy":   str(rec.get("Primary_Strategy", "")).strip(),
                "check_min":  str(rec.get("Check_Size_Min_M", "")).strip(),
                "check_max":  str(rec.get("Check_Size_Max_M", "")).strip(),
                "dm1_name":   str(rec.get("DM1_Name", "")).strip(),
                "dm1_email":  str(rec.get("DM1_Email", "")).strip(),
                "dm1_linkedin": str(rec.get("DM1_LinkedIn", "")).strip(),
                "signal":     str(rec.get("Signal_2024_2025", "")).strip(),
                "geo":        str(rec.get("Geo_Preference", "")).strip(),
                "direct_fund": str(rec.get("Direct_or_Fund", "")).strip(),
                "validation": str(rec.get("Validation", "")).strip(),
                "website":    str(rec.get("Website", "")).strip(),
                "_raw":       rec,   # full row for SQL path
            })

        self._chunks     = [r["chunk"] for r in self.records]
        self._id_to_idx  = {r["id"]: i for i, r in enumerate(self.records)}

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_models(self) -> None:
        """Load embedding model and cross-encoder (downloads on first run)."""
        self._embed_model    = SentenceTransformer(EMBED_MODEL_NAME)
        self._cross_encoder  = CrossEncoder(RERANKER_MODEL_NAME)

    def _embed_query(self, query: str) -> np.ndarray:
        """BGE requires a prefix for query embeddings (instruction-tuned)."""
        prefixed = f"Represent this sentence for searching relevant passages: {query}"
        return self._embed_model.encode([prefixed])[0]

    def _embed_docs(self, texts: list[str]) -> np.ndarray:
        """Documents are embedded without prefix."""
        return self._embed_model.encode(texts, batch_size=32)

    # ── ChromaDB setup ────────────────────────────────────────────────────────

    def _setup_chromadb(self) -> None:
        """
        Connect to (or rebuild) the ChromaDB vector store.
        If the persisted collection exists with the right count, reuse it.
        Otherwise rebuild from scratch — takes ~1 second for 200 records.
        """
        os.makedirs(DB_PATH, exist_ok=True)
        chroma = chromadb.PersistentClient(path=DB_PATH)

        try:
            coll = chroma.get_collection(COLLECTION)
            if coll.count() == len(self.records):
                self._collection = coll
                return
            chroma.delete_collection(COLLECTION)
        except Exception:
            pass

        # Rebuild
        coll = chroma.create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        embeddings = self._embed_docs(self._chunks)
        metadatas  = [
            {k: r[k] for k in
             ("fo_name", "fo_type", "country", "city", "aum_range",
              "sector", "co_invest", "esg", "strategy", "check_min",
              "check_max", "dm1_name", "dm1_email", "validation", "geo")}
            for r in self.records
        ]
        batch = 50
        for start in range(0, len(self.records), batch):
            end = min(start + batch, len(self.records))
            coll.add(
                ids        = [r["id"] for r in self.records[start:end]],
                embeddings = embeddings[start:end].tolist(),
                documents  = self._chunks[start:end],
                metadatas  = metadatas[start:end],
            )
        self._collection = coll

    # ── BM25 index ────────────────────────────────────────────────────────────

    def _build_bm25(self) -> None:
        """Build BM25 index with domain-aware stopword removal."""
        tokenised    = [self._tokenise(doc) for doc in self._chunks]
        self._bm25   = BM25Okapi(tokenised)

    def _tokenise(self, text: str) -> list[str]:
        return [
            w for w in text.lower().split()
            if w not in _FINANCE_STOPWORDS and len(w) > 2
        ]

    # ════════════════════════════════════════════════════════════════════════
    #  TECHNIQUE 1 — Hybrid Retrieval  (Vector + BM25)
    # ════════════════════════════════════════════════════════════════════════

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        vector_candidates: int = 20,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7,
        bm25_cap: float = 0.6,
        metadata_filter: Optional[dict] = None,
    ) -> list[dict]:
        """
        Hybrid retrieval: ChromaDB vector search + BM25 lexical re-rank.

        Flow:
          1. ChromaDB cosine search → top vector_candidates
          2. BM25 (domain-stopword tokenised) scored over same candidates
          3. combined = vector_weight × vec_score + bm25_weight × min(bm25, cap)
          4. Sort by combined score → top_k

        The BM25 cap (0.6) prevents keyword-flooded records from dominating.
        """
        q_emb  = self._embed_query(query)
        kwargs = {
            "query_embeddings": [q_emb.tolist()],
            "n_results":        min(vector_candidates, self._collection.count()),
            "include":          ["documents", "metadatas", "distances"],
        }
        if metadata_filter:
            kwargs["where"] = metadata_filter

        res       = self._collection.query(**kwargs)
        vec_ids   = res["ids"][0]
        vec_docs  = res["documents"][0]
        vec_metas = res["metadatas"][0]
        vec_dists = res["distances"][0]

        max_d      = max(vec_dists) + 1e-9
        vec_scores = {vid: 1 - d / max_d for vid, d in zip(vec_ids, vec_dists)}

        q_tokens   = self._tokenise(query)
        bm25_all   = self._bm25.get_scores(q_tokens)
        max_b      = max(bm25_all) + 1e-9
        bm25_scores = {
            vid: min(bm25_all[self._id_to_idx[vid]] / max_b, bm25_cap)
            for vid in vec_ids if vid in self._id_to_idx
        }

        results = []
        for vid, doc, m in zip(vec_ids, vec_docs, vec_metas):
            v_s = vec_scores.get(vid, 0)
            b_s = bm25_scores.get(vid, 0)
            results.append({
                **{k: m.get(k, "") for k in
                   ("fo_name", "country", "aum_range", "sector",
                    "dm1_name", "dm1_email", "co_invest", "esg", "strategy",
                    "check_min", "check_max", "validation", "geo")},
                "id":           vid,
                "document":     doc,
                "metadata":     m,
                "vector_score": round(v_s, 4),
                "bm25_score":   round(b_s, 4),
                "final_score":  round(vector_weight * v_s + bm25_weight * b_s, 4),
            })

        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:top_k]

    # ════════════════════════════════════════════════════════════════════════
    #  TECHNIQUE 2 — Cross-Encoder Reranking
    # ════════════════════════════════════════════════════════════════════════

    def rerank(self, query: str, docs: list[dict], top_k: int = 5) -> list[dict]:
        """
        Cross-encoder reranking: re-scores each (query, document) pair jointly.

        Unlike bi-encoders that encode query and document separately,
        cross-encoders attend over both simultaneously — much richer relevance
        signal, at the cost of not being pre-computable.

        Use after hybrid retrieval with an over-fetched candidate pool.
        Model: ms-marco-MiniLM-L-6-v2  (22M params, trained on MS MARCO)
        """
        if not docs:
            return docs
        pairs  = [(query, d["document"]) for d in docs]
        scores = self._cross_encoder.predict(pairs)
        for doc, score in zip(docs, scores):
            doc["rerank_score"] = round(float(score), 4)
        return sorted(docs, key=lambda d: d["rerank_score"], reverse=True)[:top_k]

    def retrieve_and_rerank(
        self,
        query: str,
        top_k: int = 5,
        fetch_k: int = 15,
        metadata_filter: Optional[dict] = None,
    ) -> list[dict]:
        """Two-stage: hybrid fetch_k candidates → cross-encoder narrows to top_k."""
        candidates = self.retrieve(query, top_k=fetch_k, metadata_filter=metadata_filter)
        return self.rerank(query, candidates, top_k=top_k)

    # ════════════════════════════════════════════════════════════════════════
    #  TECHNIQUE 3 — CRAG (Corrective RAG)
    # ════════════════════════════════════════════════════════════════════════

    def grade_relevance(self, query: str, doc: str) -> dict:
        """
        Grade one retrieved document's relevance to the query using Haiku.

        Returns:
            {"grade": "relevant|partially_relevant|irrelevant", "reason": "..."}

        Falls back to "relevant" on any API error so the pipeline never silently
        drops documents due to billing or rate-limit issues.
        """
        prompt = (
            "You are a relevance grader for a family office intelligence database.\n\n"
            f"USER QUERY: {query}\n\n"
            f"RETRIEVED DOCUMENT (first 600 chars):\n{doc[:600]}\n\n"
            "Grade this document:\n"
            '- "relevant"           : directly addresses the query\n'
            '- "partially_relevant" : related but incomplete\n'
            '- "irrelevant"         : does not help answer the query\n\n'
            "Return JSON only — no markdown fences:\n"
            '{"grade": "relevant|partially_relevant|irrelevant", "reason": "one sentence"}'
        )
        try:
            resp = self._client.messages.create(
                model=HAIKU_MODEL, max_tokens=100,
                messages=[{"role": "user", "content": prompt}],
            )
            return _parse_json(resp.content[0].text)
        except Exception:
            return {"grade": "relevant", "reason": "API unavailable — defaulting to relevant"}

    def corrective_retrieve(
        self,
        query: str,
        top_k: int = 5,
        fetch_k: int = 12,
        metadata_filter: Optional[dict] = None,
    ) -> tuple[list[dict], str]:
        """
        CRAG retrieval: fetch candidates → grade each → filter irrelevant.

        Returns:
            (filtered_docs, status)
            status: "all_relevant" | "partial_relevant" | "all_irrelevant"
        """
        candidates = self.retrieve_and_rerank(
            query, top_k=fetch_k, fetch_k=fetch_k + 5,
            metadata_filter=metadata_filter,
        )
        graded, relevant = [], []
        for doc in candidates:
            result = self.grade_relevance(query, doc["document"])
            doc["relevance_grade"]  = result["grade"]
            doc["relevance_reason"] = result.get("reason", "")
            graded.append(doc)
            if result["grade"] in ("relevant", "partially_relevant"):
                relevant.append(doc)

        if not relevant:
            return [], "all_irrelevant"
        status = "all_relevant" if len(relevant) == len(graded) else "partial_relevant"
        return relevant[:top_k], status

    # ════════════════════════════════════════════════════════════════════════
    #  TECHNIQUE 4 — RAG Fusion + Reciprocal Rank Fusion (RRF)
    # ════════════════════════════════════════════════════════════════════════

    def expand_query(self, query: str, n_variants: int = 3) -> list[str]:
        """
        Generate N alternative phrasings of the user query via Haiku.

        Each variant targets a different retrieval strength:
          - Formal language  → BM25 named-entity matching
          - Descriptive      → vector semantic matching
          - Keyword-dense    → BM25 exact-term matching

        Falls back to [original_query] on API errors.
        """
        prompt = (
            f"Generate {n_variants} alternative phrasings of this family office "
            f"search query.\n\nORIGINAL: {query}\n\n"
            "Rules:\n"
            "- Same intent, different vocabulary\n"
            "- Mix formal, descriptive, and keyword-dense language\n"
            "- Under 20 words each\n"
            "- Return a JSON array of strings only — no fences, no explanation\n\n"
            'Example: ["phrasing one", "phrasing two", "phrasing three"]'
        )
        try:
            resp = self._client.messages.create(
                model=HAIKU_MODEL, max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            variants = _parse_json(resp.content[0].text)
            if isinstance(variants, list):
                return [query] + [str(v) for v in variants[:n_variants]]
        except Exception:
            pass
        return [query]

    @staticmethod
    def _rrf_fuse(ranked_lists: list[list[dict]], k: int = 60) -> list[dict]:
        """
        Reciprocal Rank Fusion (Cormack et al., 2009).

        score(d) = Σ 1 / (k + rank(d))  over all lists where d appears

        k=60 is the empirically validated default (reduces top-1 dominance).
        """
        rrf_scores: dict[str, float] = {}
        all_docs_by_id: dict[str, dict] = {}

        for ranked_list in ranked_lists:
            for rank, doc in enumerate(ranked_list):
                doc_id = doc["id"]
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
                if doc_id not in all_docs_by_id:
                    all_docs_by_id[doc_id] = doc

        merged = []
        for doc_id, score in rrf_scores.items():
            doc = all_docs_by_id[doc_id].copy()
            doc["rrf_score"] = round(score, 6)
            merged.append(doc)

        merged.sort(key=lambda d: d["rrf_score"], reverse=True)
        return merged

    def fusion_retrieve(
        self,
        query: str,
        top_k: int = 5,
        n_variants: int = 3,
        metadata_filter: Optional[dict] = None,
    ) -> list[dict]:
        """
        RAG Fusion pipeline:
          1. Expand query → N+1 variants
          2. Retrieve for each variant independently
          3. Fuse with RRF (deduplication + rank aggregation)
          4. Cross-encoder rerank fused candidates
        """
        query_variants = self.expand_query(query, n_variants=n_variants)
        ranked_lists   = [
            self.retrieve(variant, top_k=10, metadata_filter=metadata_filter)
            for variant in query_variants
        ]
        fused   = self._rrf_fuse(ranked_lists)
        return self.rerank(query, fused, top_k=top_k)

    # ════════════════════════════════════════════════════════════════════════
    #  TECHNIQUE 5 — SQL Router (exact numeric/categorical filtering)
    # ════════════════════════════════════════════════════════════════════════

    def classify_query(self, query: str) -> str:
        """
        Classify query routing path.

        Returns: "numeric" | "semantic" | "hybrid"

        Numeric path handles: check size > $10M, AUM above $5B, etc.
        Semantic path handles: pure meaning-based queries.
        Hybrid: queries with both numeric constraints AND semantic content.
        """
        q = query.lower()
        has_numeric  = any(p in q for p in _NUMERIC_PATTERNS)
        semantic_wds = [w for w in q.split() if len(w) > 3 and "$" not in w]
        has_semantic = len(semantic_wds) >= 4

        if has_numeric and has_semantic:
            return "hybrid"
        if has_numeric:
            return "numeric"
        return "semantic"

    def _llm_generate_filter(self, query: str) -> Optional[str]:
        """Ask Haiku to convert a natural-language query to a pandas .query() expression."""
        prompt = (
            "Convert this natural language query into a pandas .query() expression.\n\n"
            f"Available numeric columns (float): {sorted(_SAFE_NUMERIC)}\n"
            f"Available string columns: {sorted(_SAFE_STRING)}\n\n"
            "Rules:\n"
            "1. ONLY use listed column names\n"
            "2. String comparisons: single quotes, e.g. Country == 'USA'\n"
            "3. Numeric comparisons: >, <, >=, <=\n"
            "4. Combine with 'and' / 'or'\n"
            "5. If not expressible as a column filter, respond with exactly: null\n"
            "6. Return ONLY the expression or null — no markdown, no explanation\n\n"
            "Examples:\n"
            "  'check size above $20M'         -> Check_Size_Min_M > 20\n"
            "  'US offices with check > $10M'   -> Country == 'USA' and Check_Size_Min_M >= 10\n"
            "  'ESG focused offices'            -> ESG_Focus == 'Yes'\n"
            "  'AI family offices'              -> null\n\n"
            f"QUERY: {query}"
        )
        try:
            resp = self._client.messages.create(
                model=HAIKU_MODEL, max_tokens=120,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip().strip("`").strip()
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
            return None if raw.lower() in ("null", "none", "") else raw
        except Exception:
            return None

    def _validate_filter(self, expr: str) -> bool:
        """
        Security gate: only allowlisted column names may appear as identifiers.

        Strips quoted string literals FIRST (to avoid flagging 'USA', 'Yes' etc.
        as unknown identifiers — a common pitfall with naive regex scanning).
        """
        no_strings = re.sub(r"'[^']*'", "", expr)
        no_strings = re.sub(r'"[^"]*"', "", no_strings)
        remaining  = no_strings
        for col in _ALL_SAFE_COLS:
            remaining = remaining.replace(col, "")
        unknown = set(re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', remaining)) - _ALLOWED_TOKENS
        return len(unknown) == 0

    def sql_retrieve(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Exact-match retrieval via LLM-generated pandas .query() filter.

        Why: vector search cannot handle numeric ordering.
        'check size > $10M' may retrieve $8M and $15M records equally.
        This path applies an exact DataFrame filter → 100% precise results.
        """
        filter_expr = self._llm_generate_filter(query)
        if filter_expr is None:
            return []
        if not self._validate_filter(filter_expr):
            return []
        try:
            df_copy = self.df.copy()
            for col in _SAFE_NUMERIC:
                if col in df_copy.columns:
                    df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")
            filtered = df_copy.query(filter_expr)
            results  = []
            for _, row in filtered.iterrows():
                rec_id  = str(row.get("Record_ID", "")).strip()
                matched = next((r for r in self.records if r["id"] == rec_id), None)
                if matched:
                    doc = matched.copy()
                    doc.update(
                        vector_score=0.0, bm25_score=0.0,
                        final_score=1.0, filter_expr=filter_expr,
                    )
                    results.append(doc)
            return results[:top_k]
        except Exception:
            return []

    # ════════════════════════════════════════════════════════════════════════
    #  TECHNIQUE 6 — Self-RAG Confidence Scoring
    # ════════════════════════════════════════════════════════════════════════

    def self_evaluate(self, query: str, docs: list[dict], answer: str) -> dict:
        """
        Post-generation confidence evaluation (Self-RAG inspired).

        Evaluates:
          confidence_score (1-5) : context support level
          fully_grounded  (bool) : all claims traceable to context?
          coverage_gaps   (str)  : what the dataset couldn't answer

        Falls back to neutral score 3 on API errors.
        """
        ctx = "\n".join(f"  [{d['id']}] {d['document'][:200]}" for d in docs[:5])
        prompt = (
            "Evaluate this RAG system answer.\n\n"
            f"USER QUERY: {query}\n\n"
            f"RETRIEVED CONTEXT:\n{ctx}\n\n"
            f"GENERATED ANSWER (first 400 chars):\n{answer[:400]}\n\n"
            "Return JSON only — no markdown fences:\n"
            "{\n"
            '  "confidence_score": <1-5 integer>,\n'
            '  "confidence_rationale": "<one sentence>",\n'
            '  "fully_grounded": <true or false>,\n'
            '  "grounding_issues": "<claims not in context, or none>",\n'
            '  "coverage_gaps": "<query aspects not answered, or none>",\n'
            '  "advice": "<one improvement suggestion>"\n'
            "}\n\n"
            "Scale: 5=fully answered, 4=mostly, 3=partially, 2=barely, 1=not supported"
        )
        try:
            resp = self._client.messages.create(
                model=HAIKU_MODEL, max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            return _parse_json(resp.content[0].text)
        except Exception:
            return {
                "confidence_score":     3,
                "confidence_rationale": "Evaluation unavailable",
                "fully_grounded":       True,
                "grounding_issues":     "unknown",
                "coverage_gaps":        "unknown",
                "advice":               "Add API credits for full self-evaluation",
            }

    # ════════════════════════════════════════════════════════════════════════
    #  Generation
    # ════════════════════════════════════════════════════════════════════════

    def generate(self, query: str, docs: list[dict]) -> str:
        """Generate a grounded answer from retrieved context using Claude Sonnet."""
        context = "\n\n".join(
            f"[RECORD {i}: {d['id']}]\n{d['document']}"
            for i, d in enumerate(docs, 1)
        )
        user_msg = (
            f"USER QUERY: {query}\n\n"
            f"RETRIEVED CONTEXT ({len(docs)} records):\n---\n{context}\n---\n\n"
            "Answer based ONLY on the context above."
        )
        resp = self._client.messages.create(
            model=LLM_MODEL, max_tokens=1500,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        return resp.content[0].text

    # ════════════════════════════════════════════════════════════════════════
    #  Unified Advanced Pipeline
    # ════════════════════════════════════════════════════════════════════════

    def advanced_ask(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[dict] = None,
        use_crag: bool = True,
        use_fusion: bool = True,
        use_reranking: bool = True,
        use_confidence: bool = True,
        step_callback=None,   # optional callable(step_name, detail) for UI updates
    ) -> dict:
        """
        Full 5-step advanced RAG pipeline.

        ┌─────────────────────────────────────────────────────────┐
        │  Query → Route → Retrieve (SQL + Semantic) → CRAG       │
        │         → Cross-Encoder Rerank → Generate → Self-Eval   │
        └─────────────────────────────────────────────────────────┘

        Args:
            query           : natural language query
            top_k           : final documents to return
            metadata_filter : optional ChromaDB where clause
            use_crag        : enable CRAG relevance grading
            use_fusion      : enable query expansion + RAG Fusion
            use_reranking   : enable cross-encoder reranking
            use_confidence  : enable Self-RAG confidence scoring
            step_callback   : fn(step_name, detail) called at each pipeline step

        Returns:
            dict with all pipeline outputs and metadata
        """
        def _cb(step, detail=""):
            if step_callback:
                step_callback(step, detail)

        t0 = time.perf_counter()

        # STEP 1 — Route
        route = self.classify_query(query)
        _cb("route", route)

        # STEP 2 — Retrieve
        docs: list[dict] = []

        if route in ("numeric", "hybrid"):
            sql_docs = self.sql_retrieve(query, top_k=top_k * 2)
            docs.extend(sql_docs)
            _cb("sql", f"{len(sql_docs)} records from exact filter")

        if route in ("semantic", "hybrid"):
            if use_fusion:
                sem_docs = self.fusion_retrieve(
                    query, top_k=top_k + 5,
                    metadata_filter=metadata_filter,
                )
            else:
                sem_docs = self.retrieve(
                    query, top_k=top_k + 5,
                    metadata_filter=metadata_filter,
                )
            existing = {d["id"] for d in docs}
            new_docs = [d for d in sem_docs if d["id"] not in existing]
            docs.extend(new_docs)
            _cb("semantic", f"{len(new_docs)} records from semantic search")

        _cb("retrieved", f"{len(docs)} total candidates")

        # STEP 3 — CRAG grading
        crag_status   = "skipped"
        graded_details: list[dict] = []

        if use_crag and docs:
            kept = []
            for doc in docs[:top_k + 3]:
                g = self.grade_relevance(query, doc["document"])
                doc["relevance_grade"]  = g["grade"]
                doc["relevance_reason"] = g.get("reason", "")
                graded_details.append({
                    "id":     doc["id"],
                    "name":   doc.get("fo_name", ""),
                    "grade":  g["grade"],
                    "reason": g.get("reason", ""),
                })
                if g["grade"] in ("relevant", "partially_relevant"):
                    kept.append(doc)

            _cb("crag", f"{len(kept)}/{len(docs[:top_k+3])} passed grading")

            if not kept:
                return {
                    "query": query, "docs": [], "route": route,
                    "crag_status": "all_irrelevant",
                    "answer": (
                        f"**No relevant records found** for: *{query}*\n\n"
                        "The PolarityIQ dataset of 200 family offices does not contain "
                        "sufficient information to answer this query."
                    ),
                    "confidence_score": 1,
                    "fully_grounded":   True,
                    "coverage_gaps":    "Query exceeds dataset scope",
                    "elapsed_s":        round(time.perf_counter() - t0, 2),
                    "graded_details":   graded_details,
                }

            crag_status = "all_relevant" if len(kept) == len(docs[:top_k+3]) else "partial_relevant"
            docs = kept

        # STEP 4 — Rerank
        if use_reranking and docs:
            docs = self.rerank(query, docs, top_k=top_k)
            _cb("rerank", f"cross-encoder → top {len(docs)} selected")
        else:
            docs = docs[:top_k]

        # STEP 5 — Generate
        _cb("generate", f"calling Claude Sonnet on {len(docs)} records")
        answer = self.generate(query, docs)

        # Confidence scoring
        confidence_score = None
        confidence_rationale = ""
        fully_grounded   = None
        coverage_gaps    = None
        advice           = None
        grounding_issues = None

        if use_confidence:
            ev = self.self_evaluate(query, docs, answer)
            confidence_score     = ev.get("confidence_score", 3)
            confidence_rationale = ev.get("confidence_rationale", "")
            fully_grounded       = ev.get("fully_grounded", True)
            grounding_issues     = ev.get("grounding_issues", "none")
            coverage_gaps        = ev.get("coverage_gaps", "none")
            advice               = ev.get("advice", "")
            _cb("confidence", f"score {confidence_score}/5")

        elapsed = round(time.perf_counter() - t0, 2)
        _cb("done", f"completed in {elapsed}s")

        return {
            "query":               query,
            "docs":                docs,
            "answer":              answer,
            "route":               route,
            "crag_status":         crag_status,
            "confidence_score":    confidence_score,
            "confidence_rationale": confidence_rationale,
            "fully_grounded":      fully_grounded,
            "grounding_issues":    grounding_issues,
            "coverage_gaps":       coverage_gaps,
            "advice":              advice,
            "elapsed_s":           elapsed,
            "graded_details":      graded_details if use_crag else [],
            "total_records":       len(self.records),
        }

    def ask(self, query: str, top_k: int = 5, metadata_filter: Optional[dict] = None) -> dict:
        """
        Simple pipeline (backward-compatible): hybrid retrieve → generate.
        Use advanced_ask() for the full pipeline.
        """
        docs    = self.retrieve(query, top_k=top_k, metadata_filter=metadata_filter)
        answer  = self.generate(query, docs)
        return {"query": query, "docs": docs, "answer": answer}
