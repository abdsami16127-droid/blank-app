"""Modern UI design components, light theme styling, interactive diagrams, charts, and layout helpers."""

import streamlit as st
import pandas as pd
import altair as alt
from typing import Dict, List, Optional
from src.config import (
    COLOR_PALETTE,
    SENTIMENT_NEGATIVE,
    SENTIMENT_NEUTRAL,
    SENTIMENT_POSITIVE,
)
from src.issue_extractor import IssueGroup


LIGHT_THEME_CSS = """
<style>
/* Main typography and container layout */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #0F172A;
}

/* Modern ambient gradient background */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.08) 0px, transparent 40%),
                radial-gradient(at 100% 0%, rgba(6, 182, 212, 0.07) 0px, transparent 45%),
                radial-gradient(at 50% 100%, rgba(241, 245, 249, 0.9) 0px, transparent 50%),
                #F8FAFC !important;
    background-attachment: fixed !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

[data-testid="stSidebar"] {
    background: #F8FAFC !important;
    border-right: 1px solid #E2E8F0 !important;
}

.main .block-container {
    max-width: 1200px;
    padding-top: 1.5rem;
    padding-bottom: 3.5rem;
}

/* Headings */
h1 {
    font-size: 1.95rem !important;
    font-weight: 750 !important;
    letter-spacing: -0.025em;
    color: #0F172A !important;
    margin-bottom: 0.25rem !important;
}


h2 {
    font-size: 1.35rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
    color: #1E293B !important;
    margin-top: 1.5rem !important;
    margin-bottom: 0.75rem !important;
    border-bottom: 1px solid #E2E8F0;
    padding-bottom: 0.5rem;
}

h3 {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #334155 !important;
    margin-top: 1.25rem !important;
    margin-bottom: 0.5rem !important;
}

/* Metric summary strip */
.metric-strip {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 14px;
    margin: 1.25rem 0 1.75rem 0;
}

.metric-card {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px;
    padding: 16px 18px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.metric-card:hover {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
    transform: translateY(-2px);
}

.metric-card-label {
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #475569 !important;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 6px;
}

.metric-card-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #0F172A !important;
    line-height: 1.2;
}

.metric-card-subtext {
    font-size: 0.8rem;
    color: #64748B !important;
    margin-top: 4px;
}

/* Verbatim customer quote card */
.customer-quote {
    border-left: 4px solid #3B82F6 !important;
    background: #FFFFFF !important;
    padding: 16px 20px;
    margin: 12px 0;
    border-radius: 0 8px 8px 0;
    border-top: 1px solid #E2E8F0 !important;
    border-right: 1px solid #E2E8F0 !important;
    border-bottom: 1px solid #E2E8F0 !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
}

.customer-quote-text {
    font-size: 1.0rem;
    font-style: italic;
    color: #0F172A !important;
    line-height: 1.6;
}

.customer-quote-meta {
    font-size: 0.85rem;
    color: #475569 !important;
    margin-top: 10px;
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}

/* Sentiment badges with explicit, high-contrast text */
.badge-negative {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background-color: #FEF2F2 !important;
    color: #991B1B !important;
    border: 1px solid #FECACA !important;
    padding: 4px 12px;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
}

.badge-positive {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background-color: #F0FDF4 !important;
    color: #166534 !important;
    border: 1px solid #BBF7D0 !important;
    padding: 4px 12px;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
}

.badge-neutral {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background-color: #F1F5F9 !important;
    color: #334155 !important;
    border: 1px solid #CBD5E1 !important;
    padding: 4px 12px;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
}

/* Issue topic badge with crystal clear text */
.badge-topic {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background-color: #EFF6FF !important;
    color: #1D4ED8 !important;
    border: 1px solid #BFDBFE !important;
    padding: 4px 12px;
    border-radius: 6px;
    font-size: 0.85rem;
    font-weight: 600;
}

/* Priority Level badges */
.badge-critical {
    display: inline-flex;
    align-items: center;
    background-color: #FEF2F2 !important;
    color: #991B1B !important;
    border: 1px solid #FECACA !important;
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 0.76rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}

.badge-high {
    display: inline-flex;
    align-items: center;
    background-color: #FFF7ED !important;
    color: #C2410C !important;
    border: 1px solid #FED7AA !important;
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 0.76rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}

.badge-medium {
    display: inline-flex;
    align-items: center;
    background-color: #FEFCE8 !important;
    color: #854D0E !important;
    border: 1px solid #FEF08A !important;
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 0.76rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}

.badge-low {
    display: inline-flex;
    align-items: center;
    background-color: #F1F5F9 !important;
    color: #475569 !important;
    border: 1px solid #CBD5E1 !important;
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 0.76rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}

/* Secondary info box */
.info-box {
    background-color: #F0F9FF !important;
    border: 1px solid #BAE6FD !important;
    border-left: 4px solid #0284C7 !important;
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 1.25rem;
    font-size: 0.92rem;
    color: #0369A1 !important;
}

/* Pipeline diagram flow box */
.pipeline-container {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px;
    padding: 16px;
    margin: 16px 0;
    gap: 8px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
}

.pipeline-step {
    flex: 1;
    min-width: 140px;
    background: #F8FAFC !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 6px;
    padding: 10px 12px;
    text-align: center;
}

.pipeline-step-title {
    font-size: 0.85rem;
    font-weight: 700;
    color: #0F172A !important;
    margin-top: 4px;
}

.pipeline-step-sub {
    font-size: 0.72rem;
    color: #64748B !important;
    margin-top: 2px;
}

.pipeline-arrow {
    font-size: 1.2rem;
    color: #94A3B8;
    font-weight: bold;
}

/* Visual Polarity Bar */
.polarity-bar-container {
    background: #E2E8F0 !important;
    border-radius: 8px;
    height: 26px;
    width: 100%;
    display: flex;
    overflow: hidden;
    margin: 8px 0;
    border: 1px solid #CBD5E1;
}

.polarity-bar-neg {
    background-color: #EF4444;
    color: #FFFFFF !important;
    font-size: 0.75rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
}

.polarity-bar-neu {
    background-color: #94A3B8;
    color: #FFFFFF !important;
    font-size: 0.75rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
}

.polarity-bar-pos {
    background-color: #22C55E;
    color: #FFFFFF !important;
    font-size: 0.75rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
}
</style>
"""


def apply_editorial_styles():
    """Injects responsive light theme CSS styling into the Streamlit session."""
    st.markdown(LIGHT_THEME_CSS, unsafe_allow_html=True)


def get_issue_icon(issue_name: str) -> str:
    """Returns an intuitive contextual icon for an issue category."""
    name = issue_name.lower()
    if any(k in name for k in ["battery", "charge", "drain", "power", "charging"]):
        return "🔋"
    elif any(k in name for k in ["ship", "delivery", "courier", "tracking", "late", "delay", "order", "rain"]):
        return "📦"
    elif any(k in name for k in ["app", "software", "crash", "freeze", "update", "bug", "sync"]):
        return "💻"
    elif any(k in name for k in ["bluetooth", "audio", "sound", "headphone", "speaker", "static", "crackl", "stutter"]):
        return "🎧"
    elif any(k in name for k in ["heat", "hot", "hinge", "case", "button", "build", "plastic", "hardware"]):
        return "⚙️"
    elif any(k in name for k in ["support", "refund", "billing", "service", "phone", "email", "rep"]):
        return "💬"
    elif any(k in name for k in ["quality", "great", "flawless", "best", "setup", "superb", "exceeded"]):
        return "🌟"
    else:
        return "📋"


def render_header(firebase_status: Optional[Dict] = None, user_profile: Optional[Dict] = None):
    """Renders modern executive header with custom vector SVG logo and system badges."""
    badges = []

    if user_profile:
        uname = user_profile.get("name") or user_profile.get("email", "Analyst")
        urole = user_profile.get("role", "Analyst")
        user_badge = (
            '<span style="display:inline-flex; align-items:center; gap:6px; background:#EEF2FF; '
            'border:1px solid #C7D2FE; color:#3730A3; padding:4px 12px; border-radius:9999px; '
            'font-size:0.75rem; font-weight:700; letter-spacing:0.02em;">'
            f'👤 {uname} ({urole})</span>'
        )
        badges.append(user_badge)

    if firebase_status and firebase_status.get("connected"):
        proj = firebase_status.get("project_id", "Active")
        fb_badge = (
            '<span style="display:inline-flex; align-items:center; gap:6px; background:#F0FDF4; '
            'border:1px solid #BBF7D0; color:#15803D; padding:4px 12px; border-radius:9999px; '
            'font-size:0.75rem; font-weight:700; letter-spacing:0.02em;">'
            '<span style="display:inline-block; width:7px; height:7px; border-radius:50%; background:#22C55E;"></span>'
            f'FIRESTORE CONNECTED ({proj})</span>'
        )
    else:
        fb_badge = (
            '<span style="display:inline-flex; align-items:center; gap:6px; background:#F8FAFC; '
            'border:1px solid #E2E8F0; color:#475569; padding:4px 12px; border-radius:9999px; '
            'font-size:0.75rem; font-weight:600; letter-spacing:0.02em;">'
            '<span style="display:inline-block; width:7px; height:7px; border-radius:50%; background:#94A3B8;"></span>'
            'LOCAL MODE · FIREBASE READY</span>'
        )
    badges.append(fb_badge)

    onnx_badge = (
        '<span style="display:inline-flex; align-items:center; background:#EFF6FF; border:1px solid #BFDBFE; '
        'color:#1D4ED8; padding:4px 10px; border-radius:9999px; font-size:0.75rem; font-weight:700; letter-spacing:0.02em;">'
        'LOCAL ONNX BGE-SMALL</span>'
    )
    badges.append(onnx_badge)

    badges_html = " ".join(badges)

    header_html = (
        '<div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; '
        'gap:16px; margin-bottom:1rem; padding-bottom:1.25rem; border-bottom:1px solid #E2E8F0;">'
        '<div style="display:flex; align-items:center; gap:16px;">'
        '<div style="flex-shrink:0;">'
        '<svg width="48" height="48" viewBox="0 0 44 44" fill="none" xmlns="http://www.w3.org/2000/svg" style="filter:drop-shadow(0 4px 6px rgba(79,70,229,0.15));">'
        '<defs><linearGradient id="logoGrad" x1="0" y1="0" x2="44" y2="44" gradientUnits="userSpaceOnUse">'
        '<stop offset="0%" stop-color="#4F46E5"/><stop offset="50%" stop-color="#6366F1"/><stop offset="100%" stop-color="#06B6D4"/>'
        '</linearGradient></defs>'
        '<rect width="44" height="44" rx="12" fill="url(#logoGrad)"/>'
        '<path d="M11 28V23C11 22.4477 11.4477 22 12 22H14C14.5523 22 15 22.4477 15 23V28C15 28.5523 14.5523 29 14 29H12C11.4477 29 11 28.5523 11 28Z" fill="white" fill-opacity="0.9"/>'
        '<path d="M17 28V18C17 17.4477 17.4477 17 18 17H20C20.5523 17 21 17.4477 21 18V28C21 28.5523 20.5523 29 20 29H18C17.4477 29 17 28.5523 17 28Z" fill="white" fill-opacity="0.95"/>'
        '<path d="M23 28V14C23 13.4477 23.4477 13 24 13H26C26.5523 13 27 13.4477 27 14V28C27 28.5523 26.5523 29 26 29H24C23.4477 29 23 28.5523 23 28Z" fill="white"/>'
        '<path d="M29 28V20C29 19.4477 29.4477 19 30 19H32C32.5523 19 33 19.4477 33 20V28C33 28.5523 32.5523 29 32 29H30C29.4477 29 29 28.5523 29 28Z" fill="white" fill-opacity="0.9"/>'
        '<path d="M12 20L19 14L25 11L32 16" stroke="#A5F3FC" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>'
        '<circle cx="25" cy="11" r="2.5" fill="#FFFFFF"/>'
        '</svg>'
        '</div>'
        '<div>'
        '<div style="font-size:1.65rem; font-weight:800; color:#0F172A; letter-spacing:-0.025em; line-height:1.15;">'
        'Customer Review Intelligence System</div>'
        '<div style="font-size:0.88rem; color:#475569; margin-top:3px; font-weight:500;">'
        'Enterprise Multi-Aspect NLP · Semantic Clustering · Automated Issue Prioritization</div>'
        '</div>'
        '</div>'
        f'<div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">{badges_html}</div>'
        '</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)




def render_pipeline_diagram():
    """Renders an interactive visual flowchart of the NLP intelligence pipeline."""
    html = """
    <div class="pipeline-container">
        <div class="pipeline-step">
            <div style="font-size: 1.4rem;">📥</div>
            <div class="pipeline-step-title">1. Ingestion</div>
            <div class="pipeline-step-sub">Sanitize & Validate CSV</div>
        </div>
        <div class="pipeline-arrow">➔</div>
        <div class="pipeline-step">
            <div style="font-size: 1.4rem;">⚡</div>
            <div class="pipeline-step-title">2. Sentiment</div>
            <div class="pipeline-step-sub">Calibrated VADER Engine</div>
        </div>
        <div class="pipeline-arrow">➔</div>
        <div class="pipeline-step">
            <div style="font-size: 1.4rem;">🧠</div>
            <div class="pipeline-step-title">3. Embedding</div>
            <div class="pipeline-step-sub">FastEmbed ONNX 384-d</div>
        </div>
        <div class="pipeline-arrow">➔</div>
        <div class="pipeline-step">
            <div style="font-size: 1.4rem;">🧩</div>
            <div class="pipeline-step-title">4. Clustering</div>
            <div class="pipeline-step-sub">Cosine Agglomerative</div>
        </div>
        <div class="pipeline-arrow">➔</div>
        <div class="pipeline-step">
            <div style="font-size: 1.4rem;">🏷️</div>
            <div class="pipeline-step-title">5. Topic Naming</div>
            <div class="pipeline-step-sub">c-TF-IDF & Medoid Quote</div>
        </div>
        <div class="pipeline-arrow">➔</div>
        <div class="pipeline-step">
            <div style="font-size: 1.4rem;">📊</div>
            <div class="pipeline-step-title">6. Analytics</div>
            <div class="pipeline-step-sub">Traceable Dashboard & CSV</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_polarity_bar(pos: float, neu: float, neg: float):
    """Renders a visual stacked bar for sentiment polarity components."""
    total = pos + neu + neg
    if total <= 0:
        total = 1.0
    p_neg = (neg / total) * 100
    p_neu = (neu / total) * 100
    p_pos = (pos / total) * 100

    html = f"""
    <div class="polarity-bar-container">
        <div class="polarity-bar-neg" style="width: {p_neg:.1f}%;" title="Negative: {p_neg:.1f}%">
            {f"{p_neg:.0f}%" if p_neg >= 12 else ""}
        </div>
        <div class="polarity-bar-neu" style="width: {p_neu:.1f}%;" title="Neutral: {p_neu:.1f}%">
            {f"{p_neu:.0f}%" if p_neu >= 12 else ""}
        </div>
        <div class="polarity-bar-pos" style="width: {p_pos:.1f}%;" title="Positive: {p_pos:.1f}%">
            {f"{p_pos:.0f}%" if p_pos >= 12 else ""}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_summary_strip(
    total_reviews: int,
    neg_count: int,
    neu_count: int,
    pos_count: int,
    issue_count: int,
):
    """Renders a modern metric strip with status badges and icons."""
    neg_pct = (neg_count / total_reviews * 100) if total_reviews > 0 else 0
    neu_pct = (neu_count / total_reviews * 100) if total_reviews > 0 else 0
    pos_pct = (pos_count / total_reviews * 100) if total_reviews > 0 else 0

    html = f"""
    <div class="metric-strip">
        <div class="metric-card">
            <div class="metric-card-label">📄 Total Reviews</div>
            <div class="metric-card-value">{total_reviews}</div>
            <div class="metric-card-subtext">Processed dataset</div>
        </div>
        <div class="metric-card">
            <div class="metric-card-label">🎯 Identified Issues</div>
            <div class="metric-card-value">{issue_count}</div>
            <div class="metric-card-subtext">Semantic groups</div>
        </div>
        <div class="metric-card">
            <div class="metric-card-label">🔴 Negative Complaints</div>
            <div class="metric-card-value" style="color: #B91C1C;">{neg_pct:.1f}%</div>
            <div class="metric-card-subtext">{neg_count} reviews</div>
        </div>
        <div class="metric-card">
            <div class="metric-card-label">⚪ Neutral Feedback</div>
            <div class="metric-card-value" style="color: #475569;">{neu_pct:.1f}%</div>
            <div class="metric-card-subtext">{neu_count} reviews</div>
        </div>
        <div class="metric-card">
            <div class="metric-card-label">🟢 Positive Praise</div>
            <div class="metric-card-value" style="color: #15803D;">{pos_pct:.1f}%</div>
            <div class="metric-card-subtext">{pos_count} reviews</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_quote_card(
    quote: str,
    sentiment: str,
    confidence: float,
    issue_name: str,
    review_id: Optional[int] = None,
):
    """Renders a customer quote box with metadata and badge styling."""
    badge_map = {
        SENTIMENT_NEGATIVE: ('badge-negative', '🔴 Negative'),
        SENTIMENT_POSITIVE: ('badge-positive', '🟢 Positive'),
        SENTIMENT_NEUTRAL: ('badge-neutral', '⚪ Neutral'),
    }
    badge_class, badge_label = badge_map.get(sentiment, ('badge-neutral', f'⚪ {sentiment}'))
    id_str = f"Review #{review_id} · " if review_id is not None else ""
    icon = get_issue_icon(issue_name)

    html = f"""
    <div class="customer-quote">
        <div class="customer-quote-text">"{quote}"</div>
        <div class="customer-quote-meta">
            {id_str}<span class="{badge_class}">{badge_label}</span> · Confidence: <strong>{confidence:.0%}</strong> · Issue Topic: <span class="badge-topic">{icon} {issue_name}</span>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def get_sentiment_badge_html(sentiment: str) -> str:
    """Returns HTML for inline sentiment badge with color status."""
    badge_map = {
        SENTIMENT_NEGATIVE: ('badge-negative', '🔴 Negative'),
        SENTIMENT_POSITIVE: ('badge-positive', '🟢 Positive'),
        SENTIMENT_NEUTRAL: ('badge-neutral', '⚪ Neutral'),
    }
    badge_class, badge_label = badge_map.get(sentiment, ('badge-neutral', f'⚪ {sentiment}'))
    return f'<span class="{badge_class}">{badge_label}</span>'


# ---------------------------------------------------------
# Interactive Altair Visualizations
# ---------------------------------------------------------

def render_issue_frequency_chart(issue_groups: List[IssueGroup]):
    """Renders a horizontal bar chart of issues ranked by frequency."""
    chart_data = pd.DataFrame([
        {
            "Issue Topic": f"{get_issue_icon(g.issue_name)} {g.issue_name}",
            "Review Count": g.review_count,
            "Percentage": f"{g.percentage}%",
        }
        for g in issue_groups
    ])

    chart = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color="#3B82F6")
        .encode(
            x=alt.X("Review Count:Q", title="Number of Customer Reviews"),
            y=alt.Y("Issue Topic:N", sort="-x", title=None),
            tooltip=["Issue Topic", "Review Count", "Percentage"],
        )
        .properties(
            height=max(220, len(issue_groups) * 38),
            background="transparent",
        )
        .configure_axis(
            labelFontSize=12,
            titleFontSize=13,
            labelColor="#1E293B",
            titleColor="#1E293B",
            gridColor="rgba(0, 0, 0, 0.05)",
            domainColor="rgba(0, 0, 0, 0.1)",
            tickColor="rgba(0, 0, 0, 0.1)",
            labelFont="-apple-system, BlinkMacSystemFont, Segoe UI",
        )
        .configure_view(strokeOpacity=0)
    )
    st.altair_chart(chart, use_container_width=True)


def render_sentiment_distribution_chart(neg_count: int, neu_count: int, pos_count: int):
    """Renders an interactive donut chart of sentiment proportions."""
    df_chart = pd.DataFrame([
        {"Sentiment": "Negative", "Count": neg_count, "Color": "#EF4444"},
        {"Sentiment": "Neutral", "Count": neu_count, "Color": "#94A3B8"},
        {"Sentiment": "Positive", "Count": pos_count, "Color": "#22C55E"},
    ])

    chart = (
        alt.Chart(df_chart)
        .mark_arc(innerRadius=45, stroke="#FFFFFF", strokeWidth=2)
        .encode(
            theta=alt.Theta("Count:Q"),
            color=alt.Color(
                "Sentiment:N",
                scale=alt.Scale(
                    domain=["Negative", "Neutral", "Positive"],
                    range=["#EF4444", "#94A3B8", "#22C55E"],
                ),
                legend=alt.Legend(orient="bottom", title=None, labelColor="#1E293B"),
            ),
            tooltip=["Sentiment", "Count"],
        )
        .properties(height=220, background="transparent")
        .configure_view(strokeOpacity=0)
    )
    st.altair_chart(chart, use_container_width=True)


def render_issue_sentiment_breakdown_chart(issue_groups: List[IssueGroup]):
    """Renders a stacked bar chart displaying sentiment composition per issue."""
    records = []
    for g in issue_groups:
        topic_label = f"{get_issue_icon(g.issue_name)} {g.issue_name}"
        records.append({"Issue": topic_label, "Sentiment": "Negative", "Reviews": g.sentiment_breakdown.get(SENTIMENT_NEGATIVE, 0)})
        records.append({"Issue": topic_label, "Sentiment": "Neutral", "Reviews": g.sentiment_breakdown.get(SENTIMENT_NEUTRAL, 0)})
        records.append({"Issue": topic_label, "Sentiment": "Positive", "Reviews": g.sentiment_breakdown.get(SENTIMENT_POSITIVE, 0)})

    df_stacked = pd.DataFrame(records)

    chart = (
        alt.Chart(df_stacked)
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            y=alt.Y("Issue:N", sort="-x", title=None),
            x=alt.X("Reviews:Q", stack="normalize", title="Sentiment Ratio (%)", axis=alt.Axis(format="%")),
            color=alt.Color(
                "Sentiment:N",
                scale=alt.Scale(
                    domain=["Negative", "Neutral", "Positive"],
                    range=["#EF4444", "#94A3B8", "#22C55E"],
                ),
                legend=alt.Legend(orient="top", title=None, labelColor="#1E293B"),
            ),
            tooltip=["Issue", "Sentiment", "Reviews"],
        )
        .properties(
            height=max(220, len(issue_groups) * 38),
            background="transparent",
        )
        .configure_axis(
            labelColor="#1E293B",
            titleColor="#1E293B",
            gridColor="rgba(0, 0, 0, 0.05)",
            domainColor="rgba(0, 0, 0, 0.1)",
            tickColor="rgba(0, 0, 0, 0.1)",
        )
        .configure_view(strokeOpacity=0)
    )
    st.altair_chart(chart, use_container_width=True)


def get_priority_badge_html(priority_level: str) -> str:
    """Returns clean HTML badge for issue priority level."""
    p = str(priority_level).capitalize()
    if p == "Critical":
        return '<span class="badge-critical">Critical</span>'
    elif p == "High":
        return '<span class="badge-high">High</span>'
    elif p == "Medium":
        return '<span class="badge-medium">Medium</span>'
    else:
        return '<span class="badge-low">Low</span>'


def render_performance_strip(perf_metrics: dict):
    """Renders a restrained processing performance indicator for dataset execution."""
    reviews = perf_metrics.get("total_reviews", 0)
    segments = perf_metrics.get("total_segments", 0)
    sec = perf_metrics.get("processing_time", 0.0)
    throughput = perf_metrics.get("throughput", 0.0)

    html = f"""
    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px; padding: 12px 18px; margin-bottom: 1.25rem; display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px;">
        <div>
            <span style="font-size: 0.78rem; font-weight: 600; text-transform: uppercase; color: #64748B; letter-spacing: 0.04em;">Processing Performance</span>
            <div style="font-size: 0.95rem; font-weight: 600; color: #0F172A; margin-top: 2px;">
                {reviews} reviews analyzed ({segments} aspect segments)
            </div>
        </div>
        <div style="display: flex; gap: 24px; align-items: center; flex-wrap: wrap;">
            <div>
                <span style="font-size: 0.75rem; color: #64748B;">Pipeline Time:</span>
                <span style="font-size: 0.9rem; font-weight: 700; color: #0F172A; margin-left: 4px;">{sec:.3f}s</span>
            </div>
            <div>
                <span style="font-size: 0.75rem; color: #64748B;">Throughput:</span>
                <span style="font-size: 0.9rem; font-weight: 700; color: #15803D; margin-left: 4px;">{throughput:.1f} reviews/sec</span>
            </div>
            <div>
                <span style="font-size: 0.75rem; color: #64748B;">Hardware Engine:</span>
                <span style="font-size: 0.78rem; font-weight: 600; color: #1D4ED8; background: #EFF6FF; border: 1px solid #BFDBFE; padding: 2px 8px; border-radius: 4px; margin-left: 4px;">Local CPU · ONNX Quantized</span>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

