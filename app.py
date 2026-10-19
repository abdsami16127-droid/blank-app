"""Main Streamlit application for the Customer Review Intelligence System."""

import io
from pathlib import Path
import time
from typing import List, Optional

import numpy as np
import onnxruntime  # Pre-loads Windows C++ runtime DLLs cleanly before pandas/pyarrow
import pandas as pd
import streamlit as st

from src.clustering import SemanticClusterer
from src.config import (
    COLOR_PALETTE,
    PRIORITY_CRITICAL,
    PRIORITY_HIGH,
    PRIORITY_LOW,
    PRIORITY_MEDIUM,
    PRIORITY_THRESHOLDS,
    SAMPLE_CSV_PATH,
    SENTIMENT_LABELS,
    SENTIMENT_NEGATIVE,
    SENTIMENT_NEUTRAL,
    SENTIMENT_POSITIVE,
)
from src.data_loader import (
    DataValidationError,
    create_single_review_dataframe,
    detect_review_column,
    load_reviews_from_csv,
)
from src.auth import AuthManager
from src.auth_ui import render_login_page, render_user_sidebar
from src.embeddings import EmbeddingEngine
from src.firebase_manager import FirebaseManager
from src.issue_extractor import IssueExtractor, IssueGroup
from src.segmentation import segment_dataframe, segment_review_text
from src.sentiment import SentimentAnalyzer, SentimentPrediction
from src.ui_components import (
    apply_editorial_styles,
    get_issue_icon,
    get_priority_badge_html,
    get_sentiment_badge_html,
    render_header,
    render_issue_frequency_chart,
    render_issue_sentiment_breakdown_chart,
    render_performance_strip,
    render_pipeline_diagram,
    render_polarity_bar,
    render_quote_card,
    render_sentiment_distribution_chart,
    render_summary_strip,
)

# Page configuration
st.set_page_config(
    page_title="Customer Review Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply clean light theme styling
apply_editorial_styles()


@st.cache_resource(show_spinner=False)
def get_firebase_manager() -> FirebaseManager:
    """Cached singleton instance of the Firebase Cloud manager."""
    return FirebaseManager()


@st.cache_resource(show_spinner=False)
def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Cached singleton instance of the sentiment analyzer."""
    return SentimentAnalyzer()


@st.cache_resource(show_spinner=False)
def get_embedding_engine() -> EmbeddingEngine:
    """Cached singleton instance of the ONNX embedding engine."""
    return EmbeddingEngine()


@st.cache_resource(show_spinner=False)
def get_auth_manager() -> AuthManager:
    """Cached singleton instance of the Authentication Manager."""
    return AuthManager(firebase_manager=get_firebase_manager())



@st.cache_data(show_spinner=False)
def get_knowledge_base():
    """Loads and caches sample reviews and their precomputed embeddings for nearest-match queries."""
    sample_df, _ = load_reviews_from_csv(SAMPLE_CSV_PATH)
    engine = get_embedding_engine()
    embs = engine.embed_texts(sample_df["review_text"].tolist())
    return sample_df, embs


def match_single_review(query_text: str):
    """Matches a single customer review to the closest issue and quote in the knowledge base."""
    sample_df, sample_embs = get_knowledge_base()
    engine = get_embedding_engine()
    q_emb = engine.embed_single(query_text)
    sims = np.dot(sample_embs, q_emb)
    best_idx = int(np.argmax(sims))
    best_quote = sample_df.iloc[best_idx]["review_text"]
    best_category = sample_df.iloc[best_idx].get("product_category", "General Issue")
    best_sim = float(sims[best_idx])
    return best_quote, best_category, best_sim


def run_pipeline(
    df: pd.DataFrame,
    distance_threshold: float = 0.45,
) -> tuple[pd.DataFrame, pd.DataFrame, list[IssueGroup], dict]:
    """
    Runs the complete NLP inference pipeline with multi-aspect segmentation,
    sentiment, embeddings, clustering, issue extraction, and performance profiling.
    """
    t_start = time.perf_counter()

    # 1. Multi-Aspect Segmentation: decompose reviews into distinct clauses/sentences
    segments_df = segment_dataframe(df, text_column="review_text")
    if segments_df.empty:
        segments_df = pd.DataFrame({
            "segment_id": [0],
            "review_id": [0],
            "segment_text": df["review_text"].iloc[0] if not df.empty else [""],
            "original_review_text": df["review_text"].iloc[0] if not df.empty else [""],
            "segment_index": [0],
            "total_segments_in_review": [1],
        })

    # 2. Segment-level Sentiment Analysis
    sentiment_engine = get_sentiment_analyzer()
    segments_analyzed = sentiment_engine.analyze_dataframe(segments_df, text_column="segment_text")

    # 3. Dense semantic embeddings on aspect segments
    embed_engine = get_embedding_engine()
    embeddings = embed_engine.embed_texts(segments_analyzed["segment_text"].tolist())

    # 4. Semantic clustering on aspect segments
    clusterer = SemanticClusterer(
        method="agglomerative",
        distance_threshold=distance_threshold,
    )
    labels = clusterer.fit_predict(embeddings)

    # 5. Issue extraction, prioritization & labeling on aspect segments
    extractor = IssueExtractor()
    issue_groups = extractor.extract_issues(
        segments_analyzed,
        embeddings,
        labels,
        text_column="segment_text",
    )
    enriched_segments_df = extractor.assign_issue_labels_to_dataframe(
        segments_analyzed,
        issue_groups,
        labels,
    )

    # 6. Map segment-level intelligence back to parent reviews
    priority_rank = {
        PRIORITY_CRITICAL: 4,
        PRIORITY_HIGH: 3,
        PRIORITY_MEDIUM: 2,
        PRIORITY_LOW: 1,
    }

    parent_issues_map = {}
    parent_priority_map = {}
    parent_aspect_count_map = {}
    parent_impact_map = {}

    for r_id, grp in enriched_segments_df.groupby("review_id"):
        issues_seen = list(dict.fromkeys(grp["issue_name"].tolist()))
        parent_issues_map[r_id] = " | ".join(issues_seen)
        parent_aspect_count_map[r_id] = len(grp)
        highest_p = max(
            grp["priority_level"].tolist(),
            key=lambda p: priority_rank.get(p, 0),
        )
        parent_priority_map[r_id] = highest_p
        parent_impact_map[r_id] = float(grp["impact_score"].max())

    # Holistic parent review sentiment
    df_analyzed = sentiment_engine.analyze_dataframe(df, text_column="review_text")
    final_df = df_analyzed.copy()
    final_df["assigned_issues"] = final_df["review_id"].map(parent_issues_map).fillna("Unassigned")
    final_df["issue_name"] = final_df["assigned_issues"]  # Backwards compatibility
    final_df["priority_level"] = final_df["review_id"].map(parent_priority_map).fillna(PRIORITY_LOW)
    final_df["impact_score"] = final_df["review_id"].map(parent_impact_map).fillna(0.0)
    final_df["aspect_count"] = final_df["review_id"].map(parent_aspect_count_map).fillna(1).astype(int)

    t_elapsed = max(0.0001, time.perf_counter() - t_start)
    throughput = len(df) / t_elapsed

    perf_metrics = {
        "total_reviews": len(df),
        "total_segments": len(enriched_segments_df),
        "processing_time": t_elapsed,
        "throughput": throughput,
    }

    # Sync analysis run to Firebase Cloud Firestore if connected
    curr_user = st.session_state.get("current_user")
    user_email = curr_user.get("email") if curr_user else None
    fb = get_firebase_manager()
    fb.save_analysis_run(
        dataset_name=getattr(df, "_dataset_name", "Customer Feedback Batch"),
        total_reviews=len(df),
        total_aspects=len(enriched_segments_df),
        sentiment_breakdown=df_analyzed["sentiment"].value_counts().to_dict(),
        issue_groups=issue_groups,
        execution_time=t_elapsed,
        throughput=throughput,
        user_email=user_email,
    )

    return final_df, enriched_segments_df, issue_groups, perf_metrics



# ---------------------------------------------------------
# Authentication Gate
# ---------------------------------------------------------
auth_mgr = get_auth_manager()
if not render_login_page(auth_mgr):
    st.stop()


# ---------------------------------------------------------
# Sidebar: User Profile & Ingestion Controls
# ---------------------------------------------------------
render_user_sidebar(st.session_state.get("current_user"))
st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Data Source")
input_mode = st.sidebar.radio(
    "Choose Input Method",
    options=[
        "📁 Curated Sample Dataset (61 Reviews)",
        "📤 Upload Custom CSV File",
        "✍️ Manual Review Entry (Single or Multi-line)",
    ],
    index=0,
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Clustering Sensitivity")
distance_threshold = st.sidebar.slider(
    "Distance Threshold",
    min_value=0.25,
    max_value=0.65,
    value=0.45,
    step=0.05,
    help="Controls topic specificity. Lower values split into finer micro-issues. Higher values group broader themes.",
)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "processed_df" not in st.session_state:
    st.session_state["processed_df"] = None
if "processed_segments_df" not in st.session_state:
    st.session_state["processed_segments_df"] = None
if "issue_groups" not in st.session_state:
    st.session_state["issue_groups"] = None
if "perf_metrics" not in st.session_state:
    st.session_state["perf_metrics"] = None
if "custom_labels" not in st.session_state:
    st.session_state["custom_labels"] = {}
if "validation_note" not in st.session_state:
    st.session_state["validation_note"] = ""
if "single_prediction" not in st.session_state:
    st.session_state["single_prediction"] = None
if "single_segments" not in st.session_state:
    st.session_state["single_segments"] = None

# Handle Input Modes
if input_mode == "📁 Curated Sample Dataset (61 Reviews)":
    st.sidebar.caption("61 realistic customer reviews across battery, shipping, software, audio, hardware, and support.")
    if st.sidebar.button("🚀 Analyze Sample Reviews", use_container_width=True) or st.session_state["processed_df"] is None:
        try:
            with st.spinner("Analyzing customer reviews & running semantic clustering..."):
                raw_df, report = load_reviews_from_csv(SAMPLE_CSV_PATH)
                st.session_state["validation_note"] = f"Loaded {report.valid_rows} verified customer reviews from sample dataset."
                proc_df, seg_df, groups, perf = run_pipeline(raw_df, distance_threshold=distance_threshold)
                st.session_state["processed_df"] = proc_df
                st.session_state["processed_segments_df"] = seg_df
                st.session_state["issue_groups"] = groups
                st.session_state["perf_metrics"] = perf
                st.session_state["custom_labels"] = {}
                st.session_state["single_prediction"] = None
                st.session_state["single_segments"] = None
        except Exception as e:
            st.error(f"Error loading sample dataset: {e}")

elif input_mode == "📤 Upload Custom CSV File":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            try:
                temp_df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                temp_df = pd.read_csv(uploaded_file, encoding="latin-1")
            uploaded_file.seek(0)
            col_options = list(temp_df.columns)
            detected_default = detect_review_column(col_options)
            default_idx = col_options.index(detected_default) if detected_default in col_options else 0

            selected_col = st.sidebar.selectbox(
                "Review Text Column",
                options=col_options,
                index=default_idx,
            )

            if st.sidebar.button("🚀 Process Uploaded CSV", use_container_width=True):
                with st.spinner("Parsing and analyzing uploaded file..."):
                    raw_df, report = load_reviews_from_csv(uploaded_file, explicit_column=selected_col)
                    st.session_state["validation_note"] = (
                        f"Processed {report.valid_rows} valid reviews from column '{report.detected_column}' "
                        f"({report.empty_or_null_dropped} empty rows ignored, {report.short_rows_dropped} short rows ignored)."
                    )
                    proc_df, seg_df, groups, perf = run_pipeline(raw_df, distance_threshold=distance_threshold)
                    st.session_state["processed_df"] = proc_df
                    st.session_state["processed_segments_df"] = seg_df
                    st.session_state["issue_groups"] = groups
                    st.session_state["perf_metrics"] = perf
                    st.session_state["custom_labels"] = {}
                    st.session_state["single_prediction"] = None
                    st.session_state["single_segments"] = None
        except DataValidationError as dve:
            st.sidebar.error(str(dve))
        except Exception as ex:
            st.sidebar.error(f"Failed to read file: {ex}")

elif input_mode == "✍️ Manual Review Entry (Single or Multi-line)":
    st.sidebar.caption("Enter a single review for instant deep inspection, or paste multiple reviews (one per line) for batch clustering.")

    # Preset quick-test buttons in sidebar
    st.sidebar.markdown("**Quick Example Presets:**")
    col_pre1, col_pre2 = st.sidebar.columns(2)
    default_text = "Battery is good, but the app crashes frequently and shipping took 3 weeks."
    if col_pre1.button("Preset 1 (Compound)", use_container_width=True):
        st.session_state["manual_text_val"] = "Battery is good, but the app crashes frequently and shipping took 3 weeks."
    if col_pre2.button("Preset 2 (Praise)", use_container_width=True):
        st.session_state["manual_text_val"] = "Flawless build quality and incredible audio clarity, highly recommend!"

    current_val = st.session_state.get("manual_text_val", default_text)

    manual_text = st.sidebar.text_area(
        "Review Content",
        height=130,
        value=current_val,
        placeholder="Enter one review or multiple reviews (one per line)...",
    )
    if st.sidebar.button("⚡ Analyze Review(s)", use_container_width=True):
        if manual_text.strip():
            try:
                with st.spinner("Analyzing sentiment and semantic issues..."):
                    raw_df, report = create_single_review_dataframe(manual_text)
                    st.session_state["validation_note"] = f"Analyzed {report.valid_rows} manual review(s)."

                    proc_df, seg_df, groups, perf = run_pipeline(raw_df, distance_threshold=distance_threshold)
                    st.session_state["processed_df"] = proc_df
                    st.session_state["processed_segments_df"] = seg_df
                    st.session_state["issue_groups"] = groups
                    st.session_state["perf_metrics"] = perf
                    st.session_state["custom_labels"] = {}

                    analyzer = get_sentiment_analyzer()
                    if len(raw_df) == 1:
                        pred = analyzer.predict_text(raw_df["review_text"].iloc[0])
                        st.session_state["single_prediction"] = pred
                        st.session_state["single_segments"] = seg_df
                    else:
                        st.session_state["single_prediction"] = None
                        st.session_state["single_segments"] = None
            except DataValidationError as dve:
                st.sidebar.error(str(dve))
        else:
            st.sidebar.warning("Please enter review text before submitting.")


# ---------------------------------------------------------
# Main Dashboard View
# ---------------------------------------------------------
fb_mgr = get_firebase_manager()
render_header(
    firebase_status=fb_mgr.get_status(),
    user_profile=st.session_state.get("current_user"),
)


# Collapsible AI Pipeline Architecture Diagram
with st.expander("🗺️ Click to View AI Pipeline Architecture Flowchart", expanded=False):
    render_pipeline_diagram()
    st.caption("All models run 100% locally on CPU with ONNX Runtime & calibrated VADER, requiring zero external paid APIs.")

if st.session_state["validation_note"]:
    st.markdown(f"<div class='info-box'>ℹ️ {st.session_state['validation_note']}</div>", unsafe_allow_html=True)

# Processing Performance Strip (Milestone 6 Requirement #3)
if st.session_state.get("perf_metrics"):
    render_performance_strip(st.session_state["perf_metrics"])

df_current = st.session_state["processed_df"]
seg_current = st.session_state["processed_segments_df"]
groups_current = st.session_state["issue_groups"]

# Special Display for Single Review Inspection if exactly 1 review was submitted
if st.session_state["single_prediction"] is not None and df_current is not None and len(df_current) == 1:
    pred: SentimentPrediction = st.session_state["single_prediction"]
    rev_text = df_current["review_text"].iloc[0]
    seg_single = st.session_state.get("single_segments")

    st.markdown("<h2>🔍 Single Review Deep Inspection & Multi-Aspect Polarity</h2>", unsafe_allow_html=True)

    col_s1, col_s2 = st.columns([3, 2])
    with col_s1:
        # Display verbatim quote card
        render_quote_card(
            quote=rev_text,
            sentiment=pred.label,
            confidence=pred.score,
            issue_name="Manual Query",
        )

        # Milestone 6 Multi-Aspect Decomposition Display
        if seg_single is not None and len(seg_single) > 1:
            st.markdown("### 🧩 Extracted Aspect Segments (Multi-Aspect Analysis)")
            st.caption("Compound feedback decomposed into discrete clauses so positive remarks and distinct complaints are handled independently:")
            for s_idx, s_row in seg_single.iterrows():
                badge_html = get_sentiment_badge_html(s_row["sentiment"], s_row["sentiment_score"])
                issue_badge = f"<span class='badge-topic'>{get_issue_icon(s_row['issue_name'])} {s_row['issue_name']}</span>"
                st.markdown(
                    f"<div style='background:#FFFFFF; border:1px solid #E2E8F0; border-radius:6px; padding:10px 14px; margin-bottom:8px;'>"
                    f"<div style='font-size:0.92rem; font-weight:600; color:#0F172A; margin-bottom:6px;'>“{s_row['segment_text']}”</div>"
                    f"<div style='display:flex; gap:10px; align-items:center;'>{badge_html} {issue_badge}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("### 📋 Classification Rationale & Model Notes")
        st.info(f"💡 **Model Reasoning:** {pred.summary_note}")

        # Nearest match in knowledge base
        match_quote, match_cat, match_sim = match_single_review(rev_text)
        icon = get_issue_icon(match_cat)
        st.markdown("### 🎯 Nearest Knowledge Base Match")
        st.markdown(f"- **Matched Issue Topic:** {icon} **{match_cat}**")
        st.markdown(f"- **Semantic Similarity:** `{match_sim:.1%}`")
        st.progress(max(0.0, min(1.0, match_sim)))
        st.markdown(f"- **Closest Historical Customer Quote:** *\"{match_quote}\"*")

    with col_s2:
        st.markdown("### 📊 Polarity Breakdown & Sentiment Gauges")

        sentiment_badge = "🔴 Negative" if pred.label == "Negative" else "🟢 Positive" if pred.label == "Positive" else "⚪ Neutral"
        st.metric(
            label="Holistic Predicted Sentiment Class",
            value=sentiment_badge,
            delta=f"Confidence: {pred.score:.0%}",
        )
        st.metric(
            label="Continuous Compound Valence Score",
            value=f"{pred.compound:+.3f}",
            delta="Scale: -1.0 (Critical Defect) to +1.0 (High Praise)",
        )

        st.markdown("**Visual Polarity Proportion Meter:**")
        render_polarity_bar(pos=pred.pos, neu=pred.neu, neg=pred.neg)

        col_p1, col_p2, col_p3 = st.columns(3)
        col_p1.metric("🔴 Neg Polarity", f"{pred.neg:.0%}")
        col_p2.metric("⚪ Neu Polarity", f"{pred.neu:.0%}")
        col_p3.metric("🟢 Pos Polarity", f"{pred.pos:.0%}")

    st.markdown("---")

# Full Dashboard View (Available whenever processed data exists)
if df_current is not None and groups_current is not None and not df_current.empty:
    # Apply human-in-the-loop custom label overrides if any exist
    if st.session_state["custom_labels"]:
        for group in groups_current:
            if group.cluster_id in st.session_state["custom_labels"]:
                group.issue_name = st.session_state["custom_labels"][group.cluster_id]
        cluster_map = {g.cluster_id: g.issue_name for g in groups_current}
        if seg_current is not None and "cluster_id" in seg_current.columns:
            seg_current["issue_name"] = seg_current["cluster_id"].map(cluster_map)
        # Remap parent df
        if seg_current is not None:
            parent_issues = seg_current.groupby("review_id")["issue_name"].apply(lambda s: " | ".join(dict.fromkeys(s)))
            df_current["assigned_issues"] = df_current["review_id"].map(parent_issues)
            df_current["issue_name"] = df_current["assigned_issues"]

    # 1. Executive Summary Strip
    total_revs = len(df_current)
    total_aspects = len(seg_current) if seg_current is not None else total_revs
    neg_count = int((df_current["sentiment"] == SENTIMENT_NEGATIVE).sum())
    neu_count = int((df_current["sentiment"] == SENTIMENT_NEUTRAL).sum())
    pos_count = int((df_current["sentiment"] == SENTIMENT_POSITIVE).sum())
    issue_count = len(groups_current)

    render_summary_strip(
        total_reviews=total_revs,
        neg_count=neg_count,
        neu_count=neu_count,
        pos_count=pos_count,
        issue_count=issue_count,
    )

    # 2. Main Tabs
    tab_issues, tab_explorer, tab_editor, tab_methodology, tab_firebase = st.tabs([
        "🎯 Issue Prioritization & Topics",
        "🔍 Traceable Review Explorer",
        "✏️ Label Management",
        "📖 Methodology & Architecture",
        "🔥 Cloud Backend (Firebase)",
    ])


    # ---------------- TAB 1: Identified Issues & Priority Index ----------------
    with tab_issues:
        st.markdown("<h2>🎯 Actionable Issue Priority Index</h2>", unsafe_allow_html=True)
        st.markdown(
            "Every identified issue is scored and prioritized based on **Issue Volume** and **Negative Sentiment Ratio**, "
            "highlighting where customer friction is most concentrated."
        )

        # Transparent Priority Formula & Threshold Banner
        with st.expander("ℹ️ How Priority Levels & Impact Scores Are Calculated", expanded=False):
            st.markdown(
                """
                **Impact Score Formula:**
                $$\\text{Raw Impact Score} = \\text{Issue Volume} \\times \\text{Negative Sentiment Ratio} = \\text{Total Negative Aspect Segments}$$
                $$\\text{Normalized Impact Score} = \\left( \\frac{\\text{Raw Impact Score}}{\\text{Total Customer Reviews in Dataset}} \\right) \\times 100$$

                *This answers: What percentage of our entire customer feedback volume is complaining about this specific problem?*

                **Transparent Priority Tiers:**
                * 🔴 **Critical** ($\\ge 20.0\\%$ of total feedback corpus): Severe, widespread customer defect requiring urgent escalation.
                * 🟠 **High** ($10.0\\% - 19.9\\%$ of total feedback corpus): Substantial friction impacting a major segment of users.
                * 🟡 **Medium** ($4.0\\% - 9.9\\%$ of total feedback corpus): Moderate operational issue or emerging complaint pattern.
                * ⚪ **Low** ($< 4.0\\%$ of total feedback corpus): Minor, low-volume friction or predominantly positive feedback.

                *Notice: This priority level is an automated statistical indicator derived from volume and sentiment severity. It is not an infallible business decree.*
                """
            )

        # Milestone 6 Priority Table
        priority_rows = []
        for rank, g in enumerate(groups_current, 1):
            icon = get_issue_icon(g.issue_name)
            neg = g.sentiment_breakdown.get(SENTIMENT_NEGATIVE, 0)
            neu = g.sentiment_breakdown.get(SENTIMENT_NEUTRAL, 0)
            pos = g.sentiment_breakdown.get(SENTIMENT_POSITIVE, 0)
            priority_rows.append({
                "Rank": rank,
                "Priority Level": g.priority_level,
                "Issue Topic": f"{icon} {g.issue_name}",
                "Aspect Volume": g.review_count,
                "Negative Ratio": f"{g.negative_sentiment_ratio:.1%}",
                "Impact Score (%)": g.normalized_impact_score,
                "Sentiment Breakdown": f"🔴 {neg} | ⚪ {neu} | 🟢 {pos}",
                "Representative Medoid Quote": g.representative_review,
            })

        priority_df = pd.DataFrame(priority_rows)
        st.dataframe(
            priority_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn(width="small"),
                "Priority Level": st.column_config.TextColumn("Priority", width="small"),
                "Issue Topic": st.column_config.TextColumn("Issue Topic", width="medium"),
                "Aspect Volume": st.column_config.NumberColumn("Volume", width="small"),
                "Negative Ratio": st.column_config.TextColumn("Neg. Ratio", width="small"),
                "Impact Score (%)": st.column_config.NumberColumn("Impact Score (%)", format="%.1f", width="small"),
                "Sentiment Breakdown": st.column_config.TextColumn("Breakdown", width="medium"),
                "Representative Medoid Quote": st.column_config.TextColumn("Representative Feedback", width="large"),
            },
        )

        st.markdown("---")
        st.markdown("### 📊 Semantic Issue Visual Analytics")

        # Visual Diagrams Row
        col_chart1, col_chart2 = st.columns([3, 2])
        with col_chart1:
            st.markdown("#### 📈 Issue Aspect Frequency Ranking")
            render_issue_frequency_chart(groups_current)
        with col_chart2:
            st.markdown("#### 🍩 Holistic Sentiment Distribution")
            render_sentiment_distribution_chart(neg_count, neu_count, pos_count)

        if total_revs > 1:
            st.markdown("#### 📊 Per-Issue Sentiment Breakdown")
            render_issue_sentiment_breakdown_chart(groups_current)

        # Highlight Top Critical Complaint if negative reviews exist
        neg_groups = [g for g in groups_current if g.sentiment_breakdown.get(SENTIMENT_NEGATIVE, 0) > 0]
        if neg_groups:
            top_complaint = max(neg_groups, key=lambda g: (g.normalized_impact_score, g.review_count))
            st.markdown("### 🚨 Primary Customer Complaint Spotlight", unsafe_allow_html=True)
            render_quote_card(
                quote=top_complaint.representative_review,
                sentiment=SENTIMENT_NEGATIVE,
                confidence=1.0,
                issue_name=top_complaint.issue_name,
            )

    # ---------------- TAB 2: Review Explorer ----------------
    with tab_explorer:
        st.markdown("<h2>🔍 Traceable Review Explorer</h2>", unsafe_allow_html=True)
        st.markdown("Inspect customer feedback at both the holistic parent review level and granular aspect clause level.")

        # Granularity toggle (Milestone 6 Multi-Aspect Traceability)
        col_mode, col_info = st.columns([2, 3])
        with col_mode:
            view_mode = st.radio(
                "Inspection Granularity Level",
                options=["Parent Reviews (Holistic Feedback)", "Extracted Aspects (Clause-Level Traceability)"],
                index=0,
                horizontal=True,
            )
        with col_info:
            if "Extracted Aspects" in view_mode:
                st.caption("Showing individual clauses decomposed from customer reviews, allowing independent tracking of multiple complaints.")
            else:
                st.caption("Showing original customer reviews with aggregated multi-aspect topics and highest assigned priority tier.")

        # Filter bar
        col_f1, col_f2, col_f3, col_f4 = st.columns([1, 1, 1, 2])
        with col_f1:
            sentiment_filter = st.selectbox(
                "Filter Sentiment",
                options=["All"] + SENTIMENT_LABELS,
                index=0,
            )
        with col_f2:
            issue_names_list = ["All"] + [g.issue_name for g in groups_current]
            issue_filter = st.selectbox(
                "Filter Issue Topic",
                options=issue_names_list,
                index=0,
            )
        with col_f3:
            priority_filter = st.selectbox(
                "Filter Priority",
                options=["All", PRIORITY_CRITICAL, PRIORITY_HIGH, PRIORITY_MEDIUM, PRIORITY_LOW],
                index=0,
            )
        with col_f4:
            search_query = st.text_input("Search Text", placeholder="Filter by keyword (e.g. battery, crash, shipping)...")

        # Render corresponding table based on view mode
        if "Extracted Aspects" in view_mode and seg_current is not None:
            filtered_segs = seg_current.copy()
            if sentiment_filter != "All":
                filtered_segs = filtered_segs[filtered_segs["sentiment"] == sentiment_filter]
            if issue_filter != "All":
                filtered_segs = filtered_segs[filtered_segs["issue_name"] == issue_filter]
            if priority_filter != "All":
                filtered_segs = filtered_segs[filtered_segs["priority_level"] == priority_filter]
            if search_query.strip():
                filtered_segs = filtered_segs[
                    filtered_segs["segment_text"].str.contains(search_query.strip(), case=False, na=False) |
                    filtered_segs["original_review_text"].str.contains(search_query.strip(), case=False, na=False)
                ]

            st.caption(f"Displaying {len(filtered_segs)} of {len(seg_current)} extracted aspect clauses.")

            display_cols = [
                "segment_id",
                "review_id",
                "segment_text",
                "sentiment",
                "issue_name",
                "priority_level",
                "original_review_text",
            ]
            avail_cols = [c for c in display_cols if c in filtered_segs.columns]
            styled_seg_df = filtered_segs[avail_cols].copy()
            styled_seg_df["sentiment"] = styled_seg_df["sentiment"].apply(
                lambda s: f"🔴 {s}" if s == "Negative" else f"🟢 {s}" if s == "Positive" else f"⚪ {s}"
            )
            styled_seg_df["issue_name"] = styled_seg_df["issue_name"].apply(
                lambda name: f"{get_issue_icon(name)} {name}"
            )

            st.dataframe(
                styled_seg_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "segment_id": st.column_config.NumberColumn("Seg ID", width="small"),
                    "review_id": st.column_config.NumberColumn("Parent ID", width="small"),
                    "segment_text": st.column_config.TextColumn("Extracted Aspect Clause", width="large"),
                    "sentiment": st.column_config.TextColumn("Sentiment", width="small"),
                    "issue_name": st.column_config.TextColumn("Assigned Topic", width="medium"),
                    "priority_level": st.column_config.TextColumn("Priority", width="small"),
                    "original_review_text": st.column_config.TextColumn("Full Parent Review", width="large"),
                },
            )

        else:
            # Parent reviews view
            filtered_df = df_current.copy()
            if sentiment_filter != "All":
                filtered_df = filtered_df[filtered_df["sentiment"] == sentiment_filter]
            if issue_filter != "All":
                filtered_df = filtered_df[filtered_df["assigned_issues"].str.contains(issue_filter, case=False, na=False)]
            if priority_filter != "All":
                filtered_df = filtered_df[filtered_df["priority_level"] == priority_filter]
            if search_query.strip():
                filtered_df = filtered_df[filtered_df["review_text"].str.contains(search_query.strip(), case=False, na=False)]

            st.caption(f"Displaying {len(filtered_df)} of {len(df_current)} parent reviews.")

            display_cols = [
                "review_id",
                "review_text",
                "sentiment",
                "priority_level",
                "assigned_issues",
                "aspect_count",
            ]
            avail_cols = [c for c in display_cols if c in filtered_df.columns]
            styled_table_df = filtered_df[avail_cols].copy()
            styled_table_df["sentiment"] = styled_table_df["sentiment"].apply(
                lambda s: f"🔴 {s}" if s == "Negative" else f"🟢 {s}" if s == "Positive" else f"⚪ {s}"
            )

            st.dataframe(
                styled_table_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "review_id": st.column_config.NumberColumn("ID", width="small"),
                    "review_text": st.column_config.TextColumn("Customer Review", width="large"),
                    "sentiment": st.column_config.TextColumn("Sentiment", width="small"),
                    "priority_level": st.column_config.TextColumn("Highest Priority", width="small"),
                    "assigned_issues": st.column_config.TextColumn("Assigned Issue Topics", width="large"),
                    "aspect_count": st.column_config.NumberColumn("Aspects", width="small"),
                },
            )

        # Export section
        st.markdown("---")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            csv_parent = io.StringIO()
            df_current.to_csv(csv_parent, index=False)
            st.download_button(
                label="📥 Download Parent Reviews Dataset (CSV)",
                data=csv_parent.getvalue(),
                file_name="customer_reviews_parent_export.csv",
                mime="text/csv",
                use_container_width=True,
                help="Exports parent customer reviews with assigned issues, priority tiers, and sentiment.",
            )
        with col_d2:
            if seg_current is not None:
                csv_segs = io.StringIO()
                seg_current.to_csv(csv_segs, index=False)
                st.download_button(
                    label="📥 Download Granular Aspect Segments Dataset (CSV)",
                    data=csv_segs.getvalue(),
                    file_name="customer_reviews_aspect_segments_export.csv",
                    mime="text/csv",
                    use_container_width=True,
                    help="Exports decomposed clauses with aspect topic, sentiment, priority, and parent review IDs.",
                )

    # ---------------- TAB 3: Label Management ----------------
    with tab_editor:
        st.markdown("<h2>✏️ Human-in-the-Loop Label Correction</h2>", unsafe_allow_html=True)
        st.markdown("Inspect algorithmic issue names and customize titles to match internal business naming standards.")

        with st.form("label_editor_form"):
            new_labels = {}
            for g in groups_current:
                col_cur, col_edit = st.columns([1, 2])
                with col_cur:
                    icon = get_issue_icon(g.issue_name)
                    st.markdown(f"**{icon} Group #{g.cluster_id}** ({g.review_count} aspects)")
                    st.caption(f"Top terms: {', '.join(g.top_keywords[:3]) if g.top_keywords else 'None'}")
                with col_edit:
                    new_name = st.text_input(
                        label=f"Label for Group {g.cluster_id}",
                        value=g.issue_name,
                        key=f"label_input_{g.cluster_id}",
                        label_visibility="collapsed",
                    )
                    new_labels[g.cluster_id] = new_name.strip() if new_name.strip() else g.issue_name

            save_submitted = st.form_submit_button("💾 Save and Apply Label Updates")
            if save_submitted:
                st.session_state["custom_labels"] = new_labels
                fb_mgr = get_firebase_manager()
                fb_mgr.save_custom_labels(new_labels)
                st.success("Issue labels successfully updated across the dashboard and synced to Firebase Cloud!")
                st.rerun()

    # ---------------- TAB 4: Methodology & Architecture ----------------
    with tab_methodology:
        st.markdown("<h2>📖 Model Architecture, Strengths & Limitations</h2>", unsafe_allow_html=True)

        st.markdown("### 🗺️ End-to-End Pipeline Workflow")
        render_pipeline_diagram()

        meta = SentimentAnalyzer.get_model_metadata()

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("### 🏗️ Technical Architecture")
            st.markdown(f"**Sentiment Engine:** {meta['model_name']}")
            st.markdown("**Segmentation:** Multi-aspect sentence and clause boundary parsing")
            st.markdown("**Embedding Model:** FastEmbed (ONNX Quantized BGE-Small, 384 dimensions)")
            st.markdown("**Grouping Algorithm:** Agglomerative Cosine Clustering with c-TF-IDF keyword extraction")
            st.markdown("**Prioritization Engine:** Impact Score $(Volume \\times Negative\\ Ratio)$ with transparent tier boundaries")
            st.markdown("**Cloud Backend:** Google Cloud Firestore (Audit logs & settings sync)")
            st.markdown("**Local Execution:** 100% locally runnable, CPU-optimized, zero cloud API fees.")

            st.markdown("### ✅ Architectural Strengths")
            for strength in meta["strengths"]:
                st.markdown(f"- {strength}")
            st.markdown("- **Multi-Aspect Isolation:** Mixed feedback (e.g., 'good battery, terrible app') splits into independent aspects.")
            st.markdown("- **Deterministic Priority:** Transparent mathematical formula prevents black-box opacity.")

        with col_m2:
            st.markdown("### ⚠️ Documented Limitations & Edge Cases")
            for limitation in meta["limitations"]:
                st.markdown(f"- {limitation}")
            st.markdown(
                "- **Informal or Run-on Grammar:** Reviews with run-on sentences lacking punctuation or conjunctions "
                "may be retained as single compound units."
            )
            st.markdown(
                "- **Priority as Operational Heuristic:** High-volume minor friction can mathematically rank above low-volume critical bugs; "
                "human triaging remains essential."
            )

    # ---------------- TAB 5: Cloud Backend (Firebase) ----------------
    with tab_firebase:
        st.markdown("<h2>🔥 Google Cloud Firestore Backend</h2>", unsafe_allow_html=True)
        st.markdown(
            "Persistent cloud database integration for review intelligence audits, historical run tracking, and shared team label overrides."
        )

        fb_mgr = get_firebase_manager()
        fb_stat = fb_mgr.get_status()

        col_fb1, col_fb2 = st.columns([1, 1])
        with col_fb1:
            st.markdown("### 📡 Connection Status")
            if fb_stat["connected"]:
                st.success(f"🟢 **Status:** {fb_stat['mode']}")
                st.markdown(f"- **Project ID:** `{fb_stat['project_id']}`")
                st.markdown(f"- **Database Engine:** Google Cloud Firestore")
                st.markdown(f"- **Sync Mode:** Live Cloud Synchronization")
            else:
                st.warning(f"🟡 **Status:** {fb_stat['mode']}")
                st.info(f"💡 {fb_stat['message']}")
                st.caption(
                    "The application runs 100% locally with zero errors. To enable cloud persistence, follow the setup instructions below."
                )

        with col_fb2:
            st.markdown("### 🗄️ Firestore Collections Schema")
            st.markdown(
                """
                * 📂 **`users`**: User authentication profiles (email, name, role, registration timestamp, last login).
                * 📂 **`analysis_runs`**: Stores execution audits (timestamp, user, volume, throughput, issue breakdown).
                * 📂 **`app_settings/custom_labels`**: Persists human-in-the-loop topic name overrides across browser sessions.
                * 📂 **`review_audits`**: Individual review deep-inspection records.
                """
            )

        st.markdown("---")
        st.markdown("### 📜 Recent Cloud Analysis Runs")
        history = fb_mgr.get_historical_runs(limit=5)
        if history:
            hist_rows = []
            for h in history:
                hist_rows.append({
                    "Timestamp (UTC)": str(h.get("timestamp", ""))[:19].replace("T", " "),
                    "User": h.get("user_email", "Analyst"),
                    "Dataset": h.get("dataset_name", "Curated Sample"),
                    "Reviews": h.get("total_reviews", 0),
                    "Aspects": h.get("total_aspects", 0),
                    "Issues Discovered": h.get("issue_count", 0),
                    "Throughput": f"{h.get('throughput_reviews_per_sec', 0.0)} rev/s",
                    "Execution Time": f"{h.get('execution_time_seconds', 0.0)}s",
                })
            st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)
        else:
            if fb_stat["connected"]:
                st.caption("No historical runs recorded yet. Run an analysis from the sidebar to record your first cloud audit!")
            else:
                st.caption("Cloud run logging will be visible here once Firebase credentials are added.")

        st.markdown("---")
        with st.expander("🛠️ Step-by-Step Instructions: How to Connect Your Firebase Project", expanded=not fb_stat["connected"]):
            st.markdown(
                """
                #### Step 1: Create a Free Firebase Project
                1. Navigate to the [Firebase Console](https://console.firebase.google.com/) and click **Add project**.
                2. Enter a project name (e.g. `customer-review-intel`) and create the project.

                #### Step 2: Enable Firebase Authentication
                1. In the left navigation menu, click **Build** $\\rightarrow$ **Authentication**.
                2. Click **Get started** $\\rightarrow$ Select **Email/Password** $\\rightarrow$ Enable and click **Save**.

                #### Step 3: Enable Cloud Firestore
                1. In the left navigation menu, click **Build** $\\rightarrow$ **Firestore Database**.
                2. Click **Create database**, choose a nearby region, and select **Start in test mode**.

                #### Step 4: Generate Service Account Key
                1. Click the **Project settings (Gear icon)** in the top-left $\\rightarrow$ **Service accounts** tab.
                2. Click **Generate new private key** $\\rightarrow$ **Generate key**.
                3. A JSON file will download to your computer.

                #### Step 5: Drop Into Project Root
                1. Rename the downloaded file to:
                   ```text
                   firebase_credentials.json
                   ```
                2. Place it directly inside this project folder:
                   ```text
                   C:\\Users\\abdsa\\.gemini\\antigravity\\scratch\\customer-review-intelligence\\firebase_credentials.json
                   ```
                3. Refresh this dashboard — the header badge will instantly change to **🟢 FIRESTORE CONNECTED**!
                """
            )


else:
    st.info("👈 Please select a data source from the sidebar to begin analysis.")
