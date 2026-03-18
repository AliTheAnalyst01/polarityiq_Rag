"""
PolarityIQ · Family Office Intelligence  —  app.py
====================================================
Streamlit interface for the advanced RAG pipeline.

Run locally:
    streamlit run app.py

Streamlit Cloud:
    Set ANTHROPIC_API_KEY in the app's Secrets panel.
"""

import os
import time

import streamlit as st

# ── Page config — MUST be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title="PolarityIQ · Family Office Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="metric-container"] {
    background: #0f1117;
    border: 1px solid #2d2d2d;
    border-radius: 8px;
    padding: 12px 16px;
}
.score-5 { color: #00d26a; font-weight: 700; }
.score-4 { color: #7bc8f6; font-weight: 700; }
.score-3 { color: #f5a623; font-weight: 700; }
.score-2 { color: #ff6b6b; font-weight: 700; }
.score-1 { color: #cc0000; font-weight: 700; }
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge-hybrid   { background: #2d3748; color: #7bc8f6; border: 1px solid #7bc8f6; }
.badge-semantic { background: #1a2e1a; color: #00d26a; border: 1px solid #00d26a; }
.badge-numeric  { background: #2d1f0e; color: #f5a623; border: 1px solid #f5a623; }
.badge-ok    { background: #1a2e1a; color: #00d26a; border: 1px solid #00d26a; }
.badge-warn  { background: #2d1f0e; color: #f5a623; border: 1px solid #f5a623; }
.badge-error { background: #2d0e0e; color: #ff6b6b; border: 1px solid #ff6b6b; }
.answer-card {
    border: 1px solid #2d2d2d;
    border-radius: 10px;
    padding: 24px;
    background: #0f1117;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Pipeline loader  (cached — loads only once per session)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_pipeline(api_key: str):
    from rag_pipeline import PolarityIQPipeline
    return PolarityIQPipeline(api_key=api_key)


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

CONF_BARS  = {1: "█░░░░", 2: "██░░░", 3: "███░░", 4: "████░", 5: "█████"}
CONF_EMOJI = {1: "🔴", 2: "🟠", 3: "🟡", 4: "🟢", 5: "✅"}
ROUTE_EMOJI = {"semantic": "🔍", "numeric": "🔢", "hybrid": "⚡"}
GRADE_EMOJI = {"relevant": "✅", "partially_relevant": "🟡", "irrelevant": "❌"}
CRAG_BADGE  = {
    "all_relevant":    ("ok",    "All Relevant"),
    "partial_relevant":("warn",  "Partially Relevant"),
    "all_irrelevant":  ("error", "No Relevant Records"),
    "skipped":         ("warn",  "CRAG Skipped"),
}


def _badge(text: str, style: str) -> str:
    return f'<span class="badge badge-{style}">{text}</span>'


def _conf_bar(score: int) -> str:
    bar   = CONF_BARS.get(score, "?????")
    emoji = CONF_EMOJI.get(score, "❓")
    css   = f"score-{score}"
    return f'<span class="{css}">{emoji} {bar} {score}/5</span>'


EXAMPLES = [
    "Which family offices focus on AI with check sizes above $10M?",
    "Single-family offices that made direct investments in 2024",
    "Middle East family offices open to co-investments with AUM over $5B",
    "ESG and impact investing family offices in Europe",
    "Top family offices for a $30M Series B healthcare co-investment",
    "Family offices with minimum check size above $25M",
    "Crypto and blockchain focused family offices",
    "Family offices founded after 2010 in Southeast Asia",
]


# ══════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🏦 PolarityIQ")
    st.caption("Family Office Intelligence · Advanced RAG Demo")
    st.divider()

    # API key — Streamlit secrets > env var > user input
    # st.secrets.get() raises StreamlitSecretNotFoundError if no secrets.toml exists
    # (even when calling .get with a default), so we wrap it in try/except.
    default_key = ""
    try:
        default_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        pass
    default_key = default_key or os.environ.get("ANTHROPIC_API_KEY", "")

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        value=default_key,
        help="sk-ant-... from console.anthropic.com",
    )
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    st.divider()

    # Pipeline toggles
    st.markdown("#### ⚙️ Pipeline Mode")
    advanced_mode = st.toggle(
        "Advanced RAG (CRAG + Fusion + Rerank + Self-RAG)",
        value=True,
    )
    if advanced_mode:
        c1, c2 = st.columns(2)
        use_crag       = c1.checkbox("CRAG",       value=True, help="Relevance grading")
        use_fusion     = c2.checkbox("RAG Fusion",  value=True, help="Multi-query + RRF")
        use_reranking  = c1.checkbox("Rerank",      value=True, help="Cross-encoder")
        use_confidence = c2.checkbox("Self-RAG",    value=True, help="Confidence score")
    else:
        use_crag = use_fusion = use_reranking = use_confidence = False

    st.divider()

    # Filters
    st.markdown("#### 🔎 Filters (optional)")
    top_k          = st.slider("Results returned", 3, 10, 5)
    country_filter = st.selectbox("Country", [
        "", "USA", "UK", "UAE", "Germany", "France", "Switzerland",
        "Saudi Arabia", "India", "China", "Singapore", "Brazil",
        "South Africa", "Japan", "South Korea", "Malaysia", "Australia",
    ])
    fo_type_filter  = st.selectbox("FO Type", ["", "SFO", "MFO"])
    esg_filter      = st.selectbox("ESG Focus", ["", "Yes", "No"])
    coinvest_filter = st.selectbox("Co-Investment", ["", "Yes", "No", "Selective"])

    st.divider()

    # Example queries
    st.markdown("#### 💡 Example Queries")
    for idx, ex in enumerate(EXAMPLES):
        label = ex[:50] + ("…" if len(ex) > 50 else "")
        if st.button(label, use_container_width=True, key=f"ex_{idx}"):
            st.session_state["prefill_query"] = ex

    st.divider()
    st.caption(
        "Stack: ChromaDB · BAAI/bge-small-en-v1.5 · "
        "ms-marco-MiniLM · Claude Sonnet 4.6"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Main area — Header
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    "# 🏦 PolarityIQ · Family Office Intelligence\n"
    "**200 real family offices · Natural language query · Advanced RAG pipeline**"
)

s1, s2, s3, s4 = st.columns(4)
s1.metric("Family Offices", "200")
s2.metric("Data Fields / Record", "42")
s3.metric("Embedding Dimension", "384")
s4.metric("Retrieval Method", "Hybrid + Rerank")

st.divider()

# ── Query input ────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill_query", "")
query = st.text_input(
    "Ask any question about the family offices in the dataset:",
    value=prefill,
    placeholder="e.g. Which family offices focus on AI with check sizes above $10M?",
)

col_run, col_clr = st.columns([5, 1])
run_search = col_run.button("🔍 Search", type="primary", use_container_width=True)
if col_clr.button("✕ Clear", use_container_width=True):
    st.session_state.pop("last_result", None)
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  Execute pipeline
# ══════════════════════════════════════════════════════════════════════════════

if run_search and query.strip():
    if not api_key:
        st.error("Please enter your Anthropic API key in the sidebar.")
        st.stop()

    meta_filter: dict = {}
    if country_filter:   meta_filter["country"]   = country_filter
    if fo_type_filter:   meta_filter["fo_type"]   = fo_type_filter
    if esg_filter:       meta_filter["esg"]        = esg_filter
    if coinvest_filter:  meta_filter["co_invest"]  = coinvest_filter
    meta_filter = meta_filter or None

    step_log: list[tuple[str, str]] = []

    def on_step(step: str, detail: str = "") -> None:
        step_log.append((step, detail))

    with st.status("Running Advanced RAG Pipeline…", expanded=True) as status_panel:
        status_lines = st.empty()

        def refresh_status():
            icons = {
                "route":     "🔀", "sql": "🔢", "semantic": "🔍",
                "retrieved": "📥", "crag": "🎯", "rerank": "⚡",
                "generate":  "🤖", "confidence": "📊", "done": "✅",
            }
            lines = "\n".join(
                f"{icons.get(s, '›')} **{s.title()}** — {d}"
                for s, d in step_log
            )
            status_lines.markdown(lines)

        try:
            pipe = load_pipeline(api_key)

            def on_step_with_refresh(step, detail=""):
                on_step(step, detail)
                refresh_status()

            t0 = time.perf_counter()
            if advanced_mode:
                result = pipe.advanced_ask(
                    query           = query,
                    top_k           = top_k,
                    metadata_filter = meta_filter,
                    use_crag        = use_crag,
                    use_fusion      = use_fusion,
                    use_reranking   = use_reranking,
                    use_confidence  = use_confidence,
                    step_callback   = on_step_with_refresh,
                )
            else:
                result = pipe.ask(query, top_k=top_k, metadata_filter=meta_filter)
                result.update(
                    route="semantic", crag_status="skipped",
                    confidence_score=None, fully_grounded=None,
                    coverage_gaps=None, graded_details=[],
                    elapsed_s=round(time.perf_counter() - t0, 2),
                )

            elapsed = result.get("elapsed_s", "—")
            status_panel.update(
                label=f"✅ Done in {elapsed}s  ·  {len(result.get('docs',[]))} records returned",
                state="complete",
            )
            st.session_state["last_result"] = result
            st.session_state["last_query"]  = query

        except Exception as e:
            status_panel.update(label=f"❌ Error: {e}", state="error")
            st.exception(e)
            st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  Render results
# ══════════════════════════════════════════════════════════════════════════════

result = st.session_state.get("last_result")

if result:
    docs        = result.get("docs", [])
    answer      = result.get("answer", "")
    route       = result.get("route", "semantic")
    crag_status = result.get("crag_status", "skipped")
    conf_score  = result.get("confidence_score")
    grounded    = result.get("fully_grounded")
    gaps        = result.get("coverage_gaps", "none")
    grounding_i = result.get("grounding_issues", "none")
    elapsed     = result.get("elapsed_s", "—")
    graded      = result.get("graded_details", [])
    advice      = result.get("advice", "")
    conf_text   = result.get("confidence_rationale", "")

    st.divider()

    # ── Metrics ──────────────────────────────────────────────────────────────
    st.markdown("### 📊 Pipeline Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)

    route_style = {"semantic": "semantic", "numeric": "numeric", "hybrid": "hybrid"}.get(route, "warn")
    m1.markdown(
        f"**Route**  \n{ROUTE_EMOJI.get(route,'?')} " + _badge(route.upper(), route_style),
        unsafe_allow_html=True,
    )

    m2.metric("Records Found", len(docs), help="Passed to Claude Sonnet for generation")

    crag_style, crag_label = CRAG_BADGE.get(crag_status, ("warn", crag_status))
    m3.markdown(
        "**CRAG Status**  \n" + _badge(crag_label, crag_style),
        unsafe_allow_html=True,
    )

    if conf_score is not None:
        m4.markdown(
            "**Confidence**  \n" + _conf_bar(conf_score),
            unsafe_allow_html=True,
        )
        if conf_text:
            m4.caption(conf_text[:60])
    else:
        m4.metric("Confidence", "N/A")

    if grounded is not None:
        g_label = "✅ Grounded" if grounded else "⚠️ Partial"
        g_style = "ok" if grounded else "warn"
        m5.markdown("**Grounding**  \n" + _badge(g_label, g_style), unsafe_allow_html=True)
        if grounding_i and grounding_i.lower() != "none":
            m5.caption(grounding_i[:60])
    else:
        m5.metric("Grounding", "N/A")

    st.caption(f"⏱ {elapsed}s · {result.get('total_records', 200)} records searched · model: claude-sonnet-4-6")

    # Coverage gap warning
    if gaps and gaps.lower() not in ("none", "unknown", "unknown — api unavailable"):
        st.warning(f"**Coverage Gap:** {gaps}")

    st.divider()

    # ── Answer ────────────────────────────────────────────────────────────────
    st.markdown("### 🤖 Answer")
    st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)

    if advice and advice.lower() not in ("n/a", "", "none"):
        with st.expander("💡 Improvement Advice"):
            st.info(advice)

    # ── Retrieved records ─────────────────────────────────────────────────────
    if docs:
        st.divider()
        with st.expander(f"📂 Retrieved Records ({len(docs)}) — Full Score Breakdown", expanded=False):
            for i, d in enumerate(docs, 1):
                left, right = st.columns([3, 2])

                with left:
                    st.markdown(
                        f"**#{i} — `{d.get('id','?')}` · {d.get('fo_name','?')}**  \n"
                        f"📍 {d.get('country','?')} · {d.get('aum_range','?')}  \n"
                        f"🏭 Sector: {d.get('sector','?')} · "
                        f"Strategy: {d.get('strategy','?')}  \n"
                        f"💰 Check: ${d.get('check_min','?')}M – ${d.get('check_max','?')}M · "
                        f"Co-invest: {d.get('co_invest','?')}"
                    )
                    if d.get("dm1_name"):
                        st.caption(
                            f"👤 {d.get('dm1_name','')}  "
                            + (f"· ✉️ {d.get('dm1_email','')}" if d.get("dm1_email") else "")
                        )

                    grade = d.get("relevance_grade")
                    if grade:
                        g_emoji = GRADE_EMOJI.get(grade, "❓")
                        g_style = "ok" if grade == "relevant" else "warn" if "partial" in grade else "error"
                        st.markdown(
                            f"CRAG grade: {g_emoji} " + _badge(grade.replace("_", " ").title(), g_style),
                            unsafe_allow_html=True,
                        )
                        if d.get("relevance_reason"):
                            st.caption(f"_{d['relevance_reason']}_")

                with right:
                    st.markdown("**Retrieval Score Breakdown**")
                    v_s = d.get("vector_score", 0)
                    b_s = d.get("bm25_score", 0)
                    f_s = d.get("final_score", 0)
                    r_s = d.get("rerank_score")
                    rrf = d.get("rrf_score")

                    st.progress(float(min(v_s, 1.0)), text=f"Vector score:  {v_s:.4f}")
                    st.progress(float(min(b_s, 1.0)), text=f"BM25 score:    {b_s:.4f}")
                    if r_s is not None:
                        norm = float(max(0.0, min(1.0, (r_s + 10) / 20)))
                        st.progress(norm, text=f"Rerank score:  {r_s:.4f}")
                    if rrf is not None:
                        st.progress(float(min(rrf * 100, 1.0)), text=f"RRF score:     {rrf:.6f}")
                    st.markdown(f"**Hybrid final: {f_s:.4f}**")

                with st.expander("📄 Document chunk"):
                    st.text(d.get("document", "")[:1000])

                st.divider()

    # ── CRAG grading summary ──────────────────────────────────────────────────
    if graded:
        with st.expander("🎯 CRAG Grading Details", expanded=False):
            st.caption(
                "Each candidate was graded by Claude Haiku before being passed to Claude Sonnet. "
                "Irrelevant documents are filtered out to prevent hallucination."
            )
            g_cols = st.columns(3)
            n_rel     = sum(1 for g in graded if g["grade"] == "relevant")
            n_partial = sum(1 for g in graded if g["grade"] == "partially_relevant")
            n_irrel   = sum(1 for g in graded if g["grade"] == "irrelevant")
            g_cols[0].metric("Relevant",            n_rel)
            g_cols[1].metric("Partially Relevant",  n_partial)
            g_cols[2].metric("Filtered Out",        n_irrel)
            st.divider()
            for gd in graded:
                emoji  = GRADE_EMOJI.get(gd["grade"], "❓")
                gstyle = "ok" if gd["grade"] == "relevant" else "warn" if "partial" in gd["grade"] else "error"
                st.markdown(
                    f"{emoji} **{gd['id']}** — {gd['name']}  "
                    + _badge(gd["grade"].replace("_", " ").title(), gstyle)
                    + (f"  \n_{gd.get('reason','')}_" if gd.get("reason") else ""),
                    unsafe_allow_html=True,
                )

    # ── Query history ─────────────────────────────────────────────────────────
    history = st.session_state.setdefault("history", [])
    last_q  = st.session_state.get("last_query", "")
    if not history or history[-1]["query"] != last_q:
        history.append({
            "query":      last_q,
            "docs_found": len(docs),
            "confidence": conf_score,
            "route":      route,
            "elapsed":    elapsed,
        })
        st.session_state["history"] = history[-10:]

    if len(history) > 1:
        st.divider()
        with st.expander(f"🕑 Query History ({len(history)} queries this session)"):
            for h in reversed(history):
                c_html = _conf_bar(h["confidence"]) if h["confidence"] else "N/A"
                st.markdown(
                    f"**{h['query'][:80]}**  \n"
                    f"{ROUTE_EMOJI.get(h['route'],'?')} `{h['route'].upper()}` · "
                    f"{h['docs_found']} records · Confidence: {c_html} · ⏱ {h['elapsed']}s",
                    unsafe_allow_html=True,
                )
                st.divider()

# ── Empty state ────────────────────────────────────────────────────────────────
else:
    st.info(
        "Enter a query above or click an example in the sidebar to get started.\n\n"
        "This system searches **200 real family offices** using a 5-step advanced RAG pipeline."
    )
    c1, c2, c3 = st.columns(3)
    c1.markdown(
        "#### 🔍 Hybrid Retrieval\n"
        "ChromaDB vector search + BM25 lexical matching handles both semantic "
        "queries and exact name/entity matching."
    )
    c2.markdown(
        "#### 🎯 CRAG (Corrective RAG)\n"
        "Every retrieved document is graded for relevance before the LLM sees it. "
        "Off-domain queries return honest 'not found' — no hallucinations."
    )
    c3.markdown(
        "#### 📊 Self-RAG Confidence\n"
        "After generation the system scores its own confidence (1–5), "
        "checks grounding, and flags coverage gaps."
    )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "PolarityIQ · Task #2 RAG Pipeline · "
    "ChromaDB · BAAI/bge-small-en-v1.5 · ms-marco-MiniLM-L-6-v2 · Claude Sonnet 4.6 · Streamlit"
)
