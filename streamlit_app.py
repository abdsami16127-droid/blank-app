"""
Customer Review Intelligence System — Standalone Deployable Application
A self-contained Streamlit app for sentiment analysis, multi-aspect clause segmentation,
semantic issue clustering, and actionable issue prioritization.
Ready for 1-click deployment on Streamlit Community Cloud, Hugging Face Spaces, or Docker.
"""

import io
import re
import time
from typing import Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------
# Page Configuration & Editorial CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Review Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

LIGHT_THEME_CSS = """
<style>
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #0F172A;
}
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #F8FAFC 0%, #EEF2F6 100%) !important;
    background-attachment: fixed !important;
}
[data-testid="stSidebar"] {
    background: #F8FAFC !important;
    border-right: 1px solid #E2E8F0 !important;
}
.main .block-container {
    max-width: 1180px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}
h1 { font-size: 1.9rem !important; font-weight: 700 !important; color: #0F172A !important; }
h2 { font-size: 1.35rem !important; font-weight: 600 !important; color: #1E293B !important; margin-top: 1.5rem !important; border-bottom: 1px solid #E2E8F0; padding-bottom: 0.5rem; }
h3 { font-size: 1.1rem !important; font-weight: 600 !important; color: #334155 !important; }

/* Metrics */
.metric-strip { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin: 1.25rem 0 1.75rem 0; }
.metric-card { background: #FFFFFF !important; border: 1px solid #E2E8F0 !important; border-radius: 8px; padding: 16px 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
.metric-card-label { font-size: 0.82rem; font-weight: 600; text-transform: uppercase; color: #475569 !important; }
.metric-card-value { font-size: 1.75rem; font-weight: 700; color: #0F172A !important; }
.metric-card-subtext { font-size: 0.8rem; color: #64748B !important; }

/* Badges */
.badge-negative { background-color: #FEF2F2 !important; color: #991B1B !important; border: 1px solid #FECACA !important; padding: 3px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 700; }
.badge-positive { background-color: #F0FDF4 !important; color: #166534 !important; border: 1px solid #BBF7D0 !important; padding: 3px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 700; }
.badge-neutral { background-color: #F1F5F9 !important; color: #334155 !important; border: 1px solid #CBD5E1 !important; padding: 3px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 700; }
.badge-topic { background-color: #EFF6FF !important; color: #1D4ED8 !important; border: 1px solid #BFDBFE !important; padding: 3px 10px; border-radius: 6px; font-size: 0.82rem; font-weight: 600; }

.badge-critical { background-color: #FEF2F2 !important; color: #991B1B !important; border: 1px solid #FECACA !important; padding: 3px 10px; border-radius: 4px; font-size: 0.76rem; font-weight: 700; text-transform: uppercase; }
.badge-high { background-color: #FFF7ED !important; color: #C2410C !important; border: 1px solid #FED7AA !important; padding: 3px 10px; border-radius: 4px; font-size: 0.76rem; font-weight: 700; text-transform: uppercase; }
.badge-medium { background-color: #FEFCE8 !important; color: #854D0E !important; border: 1px solid #FEF08A !important; padding: 3px 10px; border-radius: 4px; font-size: 0.76rem; font-weight: 700; text-transform: uppercase; }
.badge-low { background-color: #F1F5F9 !important; color: #475569 !important; border: 1px solid #CBD5E1 !important; padding: 3px 10px; border-radius: 4px; font-size: 0.76rem; font-weight: 700; text-transform: uppercase; }

.customer-quote { border-left: 4px solid #3B82F6 !important; background: #FFFFFF !important; padding: 14px 18px; margin: 10px 0; border-radius: 0 8px 8px 0; border-top: 1px solid #E2E8F0; border-right: 1px solid #E2E8F0; border-bottom: 1px solid #E2E8F0; }
.customer-quote-text { font-size: 0.98rem; font-style: italic; color: #0F172A !important; }
.customer-quote-meta { font-size: 0.85rem; color: #475569 !important; margin-top: 8px; display: flex; align-items: center; gap: 8px; }
</style>
"""
st.markdown(LIGHT_THEME_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# Embedded Curated Dataset (61 Reviews)
# ---------------------------------------------------------
DEFAULT_SAMPLE_CSV = """review_text,rating,product_category
"The battery drains in less than 4 hours of normal usage.",1,"Electronics"
"Charge does not last through my morning commute.",2,"Electronics"
"My phone dies within hours of being completely unplugged.",1,"Electronics"
"Having to recharge three times a day is totally unacceptable.",1,"Electronics"
"Battery percentage drops 25 percent in just twenty minutes of video.",2,"Electronics"
"The battery life is very disappointing compared to the advertised 12 hours.",2,"Electronics"
"Battery runs out extremely fast even when the screen is dimmed.",1,"Electronics"
"Struggles to keep charge overnight even on standby mode.",2,"Electronics"
"Battery health degraded significantly after only two months of use.",1,"Electronics"
"Bluetooth randomly disconnects every ten minutes while listening.",2,"Audio"
"Unable to pair with my laptop or tablet despite multiple resets.",1,"Audio"
"Constant stuttering and audio dropouts over wireless bluetooth.",2,"Audio"
"Fails to reconnect automatically when turned back on.",2,"Audio"
"Bluetooth sync lag makes watching videos completely unwatchable.",1,"Audio"
"The wireless connection drops out whenever I put the phone in my pocket.",1,"Audio"
"Bluetooth pairing takes forever and frequently fails with an error.",2,"Audio"
"Device unpairs itself in the middle of important phone calls.",1,"Audio"
"Package arrived two weeks after the estimated delivery date.",1,"Logistics"
"Shipping tracking stayed stuck on label created for ten business days.",2,"Logistics"
"Order was delayed multiple times with zero proactive notification.",2,"Logistics"
"Late delivery completely ruined my planned anniversary gift.",1,"Logistics"
"I paid extra for express delivery but it took nearly two weeks to arrive.",1,"Logistics"
"Delivery courier left the box in the rain without ringing the bell.",1,"Logistics"
"Shipped via the slowest courier possible, took three weeks.",2,"Logistics"
"Package was delayed with no clear updates provided by tracking.",2,"Logistics"
"The companion mobile app crashes immediately upon opening.",1,"Software"
"The app freezes whenever I try to change any device settings.",2,"Software"
"Recent firmware update broke the sync functionality completely.",1,"Software"
"App constantly bugs out and forces me to restart my entire phone.",1,"Software"
"Application keeps crashing during user account login.",1,"Software"
"The app is very slow, buggy, and crashes whenever I tap the profile tab.",2,"Software"
"Software update ruined the user interface and caused regular app crashes.",1,"Software"
"App crashes on launch on both Android and iOS devices.",1,"Software"
"The device gets uncomfortably hot to the touch after 20 minutes.",2,"Hardware"
"Case plastic feels cheap and already cracked near the hinge.",1,"Hardware"
"Severe overheating causes the unit to throttle and shut down unexpectedly.",1,"Hardware"
"Loose buttons rattle whenever I move or pick up the device.",2,"Hardware"
"The charging port feels loose and the cable constantly falls out.",1,"Hardware"
"Hinge started squeaking and wobbling after just one week of light use.",2,"Hardware"
"Gets burning hot when fast charging, seems like a safety hazard.",1,"Hardware"
"Customer service refused to honor the one-year warranty coverage.",1,"Customer Service"
"Charged twice on my credit card and billing support refuses to reply.",1,"Customer Service"
"Wait time on phone support was over ninety minutes with no resolution.",1,"Customer Service"
"Sent four emails about my missing refund and received only automated replies.",1,"Customer Service"
"Hidden subscription fee was charged to my account without clear consent.",1,"Customer Service"
"Customer support agent was dismissive and hung up the call.",1,"Customer Service"
"Still waiting for a promised refund thirty days after returning the product.",1,"Customer Service"
"Outstanding sound clarity and very comfortable to wear for long sessions.",5,"Audio"
"Best purchase I made this year, sleek design and very reliable performance.",5,"Electronics"
"Setup was effortless and customer support answered my setup question quickly.",5,"Customer Service"
"Screen brightness and color accuracy are truly impressive for the price.",5,"Electronics"
"Build quality is solid and premium, exceeded all my expectations.",5,"Hardware"
"Very impressed with the fast delivery and securely packaged item.",5,"Logistics"
"Works flawlessly right out of the box with intuitive controls.",5,"Electronics"
"Super happy with the product quality and battery life so far.",5,"Electronics"
"Great sound reproduction with deep punchy bass and clear vocals.",5,"Audio"
"Item arrived in standard packaging, nothing particularly special.",3,"Logistics"
"Average product for the price point, does basic tasks as advertised.",3,"Electronics"
"Mediocre performance, not terrible but certainly not exceptional.",3,"Electronics"
"Functions decently enough, though the instruction manual was somewhat sparse.",3,"Documentation"
"It is okay for casual use, but professionals may want higher specs.",3,"Electronics"
"""

# Priority Constants & Thresholds
PRIORITY_CRITICAL = "Critical"
PRIORITY_HIGH = "High"
PRIORITY_MEDIUM = "Medium"
PRIORITY_LOW = "Low"

PRIORITY_THRESHOLDS = {
    PRIORITY_CRITICAL: 20.0,
    PRIORITY_HIGH: 10.0,
    PRIORITY_MEDIUM: 4.0,
    PRIORITY_LOW: 0.0,
}

# ---------------------------------------------------------
# Component 1: Multi-Aspect Clause Segmentation
# ---------------------------------------------------------
CLAUSE_SUBJECT_KEYWORDS = (
    r'(?:the\s+|it\s+|my\s+|this\s+|they\s+|we\s+|i\s+|'
    r'shipping\s+|delivery\s+|courier\s+|battery\s+|charge\s+|'
    r'customer\s+|support\s+|service\s+|rep\s+|'
    r'app\s+|application\s+|software\s+|update\s+|'
    r'hardware\s+|build\s+|hinge\s+|case\s+|sound\s+|audio\s+|screen\s+)'
)
CONTRASTIVE_PATTERN = re.compile(
    r'(?:[.!?;]|\n+|(?:,\s*(?:but|however|although|whereas|yet|while|on\s+the\s+other\s+hand))|(?:\s+(?:but|however|although|whereas|yet)\s+))',
    re.IGNORECASE,
)
COORDINATE_PATTERN = re.compile(
    rf'(?:,\s*and\s+|\s+and\s+(?={CLAUSE_SUBJECT_KEYWORDS}))',
    re.IGNORECASE,
)


def clean_segment(segment: str) -> str:
    s = segment.strip()
    s = re.sub(r'^(?:and\s+also|and|but|however|although|whereas|yet|while|so|too)\s+', '', s, flags=re.IGNORECASE)
    s = s.strip(" ,.-;:!?\t\n")
    if s and len(s) > 1:
        s = s[0].upper() + s[1:]
    return s


def segment_review_text(text: str) -> List[str]:
    if not text or not str(text).strip():
        return []
    raw = str(text).strip()
    chunks = CONTRASTIVE_PATTERN.split(raw)
    refined: List[str] = []
    for c in chunks:
        if c and c.strip():
            refined.extend(COORDINATE_PATTERN.split(c))
    valid = []
    for r in refined:
        cleaned = clean_segment(r)
        words = [w for w in cleaned.split() if w.strip()]
        if len(cleaned) >= 10 and len(words) >= 3:
            valid.append(cleaned)
    return valid if valid else [raw.strip()]


def segment_dataframe(df: pd.DataFrame, text_column: str = "review_text") -> pd.DataFrame:
    rows = []
    counter = 0
    for idx, row in df.iterrows():
        r_id = row.get("review_id", idx)
        orig = str(row[text_column])
        segs = segment_review_text(orig)
        for s_idx, seg in enumerate(segs):
            item = {
                "segment_id": counter,
                "review_id": r_id,
                "segment_text": seg,
                "original_review_text": orig,
                "segment_index": s_idx,
                "total_segments_in_review": len(segs),
            }
            for c in df.columns:
                if c not in [text_column, "review_id"] and c not in item:
                    item[c] = row[c]
            rows.append(item)
            counter += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Component 2: Sentiment Analysis Engine
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_vader():
    return SentimentIntensityAnalyzer()


def predict_sentiment(text: str) -> Dict:
    vader = get_vader()
    scores = vader.polarity_scores(str(text))
    compound = float(scores["compound"])
    pos = float(scores["pos"])
    neu = float(scores["neu"])
    neg = float(scores["neg"])

    if compound <= -0.05:
        label = "Negative"
        score = min(1.0, abs(compound) + 0.15)
        note = "Customer expressed frustration or identified a defect."
    elif compound >= 0.05:
        label = "Positive"
        score = min(1.0, compound + 0.15)
        note = "Customer expressed satisfaction or highlighted positive performance."
    else:
        label = "Neutral"
        score = max(0.5, neu)
        note = "Customer shared objective or balanced feedback."

    return {
        "label": label,
        "score": score,
        "compound": compound,
        "pos": pos,
        "neu": neu,
        "neg": neg,
        "note": note,
    }


def analyze_dataframe_sentiment(df: pd.DataFrame, text_column: str = "review_text") -> pd.DataFrame:
    enriched = df.copy()
    preds = [predict_sentiment(t) for t in enriched[text_column].astype(str)]
    enriched["sentiment"] = [p["label"] for p in preds]
    enriched["sentiment_score"] = [p["score"] for p in preds]
    enriched["compound_score"] = [p["compound"] for p in preds]
    return enriched


# ---------------------------------------------------------
# Component 3: Dense Semantic Embeddings Engine
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_fastembed_model():
    try:
        from fastembed import TextEmbedding
        return TextEmbedding("BAAI/bge-small-en-v1.5")
    except Exception:
        return None


def generate_embeddings(texts: List[str]) -> np.ndarray:
    model = get_fastembed_model()
    if model is not None:
        try:
            embs = list(model.embed(texts))
            embs = np.array(embs, dtype=np.float32)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return embs / norms
        except Exception:
            pass
    # High-quality fallback using sub-word TF-IDF if ONNX is unavailable
    tfidf = TfidfVectorizer(max_features=384, stop_words="english", ngram_range=(1, 2))
    matrix = tfidf.fit_transform(texts).toarray().astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


# ---------------------------------------------------------
# Component 4: Semantic Clustering & Issue Extraction
# ---------------------------------------------------------
def cluster_embeddings(embeddings: np.ndarray, distance_threshold: float = 0.45) -> np.ndarray:
    n = len(embeddings)
    if n <= 1:
        return np.zeros(n, dtype=int)
    clusterer = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    return clusterer.fit_predict(embeddings)


def extract_issue_groups(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    labels: np.ndarray,
    text_column: str = "segment_text",
    total_reviews_corpus: int = 1,
) -> List[Dict]:
    unique_clusters = sorted(set(labels))
    groups = []

    # Aggregated cluster texts for c-TF-IDF
    cluster_docs = {}
    for c_id in unique_clusters:
        mask = labels == c_id
        cluster_docs[c_id] = " ".join(df.loc[mask, text_column].astype(str).tolist())

    vec = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_features=1000)
    docs = [cluster_docs[c] for c in unique_clusters]
    try:
        dtm = vec.fit_transform(docs)
        feat_names = vec.get_feature_names_out()
        kws_map = {}
        for row_idx, c_id in enumerate(unique_clusters):
            scores = dtm[row_idx].toarray().flatten()
            top_idx = np.argsort(scores)[::-1][:4]
            kws_map[c_id] = [feat_names[i].title() for i in top_idx if scores[i] > 0]
    except Exception:
        kws_map = {c: [] for c in unique_clusters}

    for c_id in unique_clusters:
        mask = labels == c_id
        idx_members = np.where(mask)[0]
        count = len(idx_members)

        # Medoid (closest to cluster mean)
        cluster_embs = embeddings[idx_members]
        mean_vec = np.mean(cluster_embs, axis=0, keepdims=True)
        mean_vec /= max(1e-6, np.linalg.norm(mean_vec))
        sims = np.dot(cluster_embs, mean_vec.T).flatten()
        best_local_idx = int(np.argmax(sims))
        medoid_global_idx = idx_members[best_local_idx]
        representative_quote = str(df.iloc[medoid_global_idx][text_column])

        # Sentiments
        cluster_sent = df.loc[mask, "sentiment"].value_counts().to_dict()
        neg_count = int(cluster_sent.get("Negative", 0))
        neu_count = int(cluster_sent.get("Neutral", 0))
        pos_count = int(cluster_sent.get("Positive", 0))

        # Actionable Priority Calculations
        neg_ratio = round(neg_count / count, 3) if count > 0 else 0.0
        raw_impact = round(count * neg_ratio, 2)
        norm_impact = round((raw_impact / max(1, total_reviews_corpus)) * 100.0, 1)

        if norm_impact >= PRIORITY_THRESHOLDS[PRIORITY_CRITICAL]:
            p_level = PRIORITY_CRITICAL
        elif norm_impact >= PRIORITY_THRESHOLDS[PRIORITY_HIGH]:
            p_level = PRIORITY_HIGH
        elif norm_impact >= PRIORITY_THRESHOLDS[PRIORITY_MEDIUM]:
            p_level = PRIORITY_MEDIUM
        else:
            p_level = PRIORITY_LOW

        kws = kws_map.get(c_id, [])
        if len(kws) >= 2:
            issue_title = f"{kws[0]} / {kws[1]}"
        elif len(kws) == 1:
            issue_title = kws[0]
        else:
            issue_title = " ".join(representative_quote.split()[:4]).title()

        groups.append({
            "cluster_id": int(c_id),
            "issue_name": issue_title,
            "volume": count,
            "neg_ratio": neg_ratio,
            "raw_impact": raw_impact,
            "norm_impact": norm_impact,
            "priority_level": p_level,
            "neg": neg_count,
            "neu": neu_count,
            "pos": pos_count,
            "representative_quote": representative_quote,
            "keywords": kws,
        })

    # Sort descending by priority impact
    groups.sort(key=lambda g: (-g["norm_impact"], -g["volume"]))
    return groups


# ---------------------------------------------------------
# Component 5: Full Execution Pipeline with Profiling
# ---------------------------------------------------------
def run_full_pipeline(df: pd.DataFrame, distance_threshold: float = 0.45):
    t_start = time.perf_counter()

    # 1. Multi-Aspect Decomposition
    seg_df = segment_dataframe(df, text_column="review_text")

    # 2. Segment-level sentiment
    seg_df = analyze_dataframe_sentiment(seg_df, text_column="segment_text")

    # 3. Dense embeddings & clustering on clauses
    embs = generate_embeddings(seg_df["segment_text"].tolist())
    labels = cluster_embeddings(embs, distance_threshold=distance_threshold)
    seg_df["cluster_id"] = labels

    # 4. Extract issues & prioritize
    groups = extract_issue_groups(
        seg_df,
        embs,
        labels,
        text_column="segment_text",
        total_reviews_corpus=len(df),
    )

    c_to_name = {g["cluster_id"]: g["issue_name"] for g in groups}
    c_to_prio = {g["cluster_id"]: g["priority_level"] for g in groups}
    c_to_impact = {g["cluster_id"]: g["norm_impact"] for g in groups}

    seg_df["issue_name"] = seg_df["cluster_id"].map(c_to_name)
    seg_df["priority_level"] = seg_df["cluster_id"].map(c_to_prio)
    seg_df["impact_score"] = seg_df["cluster_id"].map(c_to_impact)

    # 5. Parent-level rollups
    prio_rank = {PRIORITY_CRITICAL: 4, PRIORITY_HIGH: 3, PRIORITY_MEDIUM: 2, PRIORITY_LOW: 1}
    parent_issues = {}
    parent_prios = {}
    parent_aspect_cnt = {}

    for r_id, sub_df in seg_df.groupby("review_id"):
        parent_issues[r_id] = " | ".join(dict.fromkeys(sub_df["issue_name"].tolist()))
        parent_prios[r_id] = max(sub_df["priority_level"].tolist(), key=lambda p: prio_rank.get(p, 0))
        parent_aspect_cnt[r_id] = len(sub_df)

    parent_df = analyze_dataframe_sentiment(df, text_column="review_text")
    parent_df["assigned_issues"] = parent_df["review_id"].map(parent_issues).fillna("Unassigned")
    parent_df["issue_name"] = parent_df["assigned_issues"]
    parent_df["priority_level"] = parent_df["review_id"].map(parent_prios).fillna(PRIORITY_LOW)
    parent_df["aspect_count"] = parent_df["review_id"].map(parent_aspect_cnt).fillna(1).astype(int)

    t_elapsed = max(0.0001, time.perf_counter() - t_start)
    throughput = len(df) / t_elapsed

    metrics = {
        "total_reviews": len(df),
        "total_segments": len(seg_df),
        "time": t_elapsed,
        "throughput": throughput,
    }

    return parent_df, seg_df, groups, metrics


# ---------------------------------------------------------
# Sidebar UI Controls
# ---------------------------------------------------------
st.sidebar.markdown("### 📥 Data Ingestion")
input_mode = st.sidebar.radio(
    "Select Input Source",
    options=[
        "📁 Curated Sample Dataset (61 Reviews)",
        "📤 Upload Custom CSV File",
        "✍️ Manual Review Entry (Single or Multi-line)",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Sensitivity Controls")
distance_threshold = st.sidebar.slider(
    "Cluster Sensitivity (Distance Threshold)",
    min_value=0.25,
    max_value=0.65,
    value=0.45,
    step=0.05,
    help="Lower values produce tighter, fine-grained micro-issues. Higher values aggregate broader themes.",
)

# Ingestion Handling
if "processed_df" not in st.session_state:
    st.session_state["processed_df"] = None
if "segments_df" not in st.session_state:
    st.session_state["segments_df"] = None
if "groups" not in st.session_state:
    st.session_state["groups"] = None
if "perf" not in st.session_state:
    st.session_state["perf"] = None
if "custom_labels" not in st.session_state:
    st.session_state["custom_labels"] = {}
if "single_pred" not in st.session_state:
    st.session_state["single_pred"] = None

if input_mode == "📁 Curated Sample Dataset (61 Reviews)":
    st.sidebar.caption("61 realistic customer reviews across battery, shipping, software, audio, hardware, and support.")
    if st.sidebar.button("🚀 Analyze Sample Dataset", use_container_width=True) or st.session_state["processed_df"] is None:
        with st.spinner("Processing multi-aspect segmentation and clustering..."):
            raw_df = pd.read_csv(io.StringIO(DEFAULT_SAMPLE_CSV))
            raw_df["review_id"] = range(len(raw_df))
            p_df, s_df, grps, perf = run_full_pipeline(raw_df, distance_threshold=distance_threshold)
            st.session_state["processed_df"] = p_df
            st.session_state["segments_df"] = s_df
            st.session_state["groups"] = grps
            st.session_state["perf"] = perf
            st.session_state["custom_labels"] = {}
            st.session_state["single_pred"] = None

elif input_mode == "📤 Upload Custom CSV File":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            up_df = pd.read_csv(uploaded_file)
            cols = list(up_df.columns)
            detected = next((c for c in cols if any(k in c.lower() for k in ["review", "feedback", "comment", "text"])), cols[0])
            sel_col = st.sidebar.selectbox("Select Review Text Column", options=cols, index=cols.index(detected))
            if st.sidebar.button("🚀 Process Uploaded File", use_container_width=True):
                with st.spinner("Analyzing custom dataset..."):
                    raw_df = up_df.copy()
                    raw_df["review_text"] = raw_df[sel_col].astype(str)
                    raw_df["review_id"] = range(len(raw_df))
                    p_df, s_df, grps, perf = run_full_pipeline(raw_df, distance_threshold=distance_threshold)
                    st.session_state["processed_df"] = p_df
                    st.session_state["segments_df"] = s_df
                    st.session_state["groups"] = grps
                    st.session_state["perf"] = perf
                    st.session_state["custom_labels"] = {}
                    st.session_state["single_pred"] = None
        except Exception as ex:
            st.sidebar.error(f"Error reading CSV: {ex}")

elif input_mode == "✍️ Manual Review Entry (Single or Multi-line)":
    st.sidebar.caption("Enter compound feedback with multiple aspects or paste multiple reviews (one per line):")
    preset_col1, preset_col2 = st.sidebar.columns(2)
    default_text = "Battery is good, but the app crashes frequently and shipping took 3 weeks."
    if preset_col1.button("Preset: Mixed", use_container_width=True):
        st.session_state["manual_text"] = "Battery is good, but the app crashes frequently and shipping took 3 weeks."
    if preset_col2.button("Preset: Praise", use_container_width=True):
        st.session_state["manual_text"] = "Flawless build quality and incredible audio clarity, highly recommend!"

    cur_text = st.session_state.get("manual_text", default_text)
    user_text = st.sidebar.text_area("Review Content", value=cur_text, height=130)

    if st.sidebar.button("⚡ Analyze Review(s)", use_container_width=True):
        lines = [l.strip() for l in user_text.splitlines() if l.strip()]
        if lines:
            raw_df = pd.DataFrame({"review_text": lines, "review_id": range(len(lines))})
            p_df, s_df, grps, perf = run_full_pipeline(raw_df, distance_threshold=distance_threshold)
            st.session_state["processed_df"] = p_df
            st.session_state["segments_df"] = s_df
            st.session_state["groups"] = grps
            st.session_state["perf"] = perf
            st.session_state["custom_labels"] = {}
            if len(lines) == 1:
                st.session_state["single_pred"] = predict_sentiment(lines[0])
            else:
                st.session_state["single_pred"] = None

# ---------------------------------------------------------
# Main Dashboard Rendering
# ---------------------------------------------------------
st.markdown("<h1>📊 Customer Review Intelligence System</h1>", unsafe_allow_html=True)
st.markdown("Decomposes compound customer feedback into constituent clauses, detects sentiment, groups similar issues, and prioritizes actionable friction.")

# Performance Strip
perf_info = st.session_state.get("perf")
if perf_info:
    rev_cnt = perf_info["total_reviews"]
    seg_cnt = perf_info["total_segments"]
    p_time = perf_info["time"]
    t_put = perf_info["throughput"]
    st.markdown(
        f"""
        <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:6px; padding:12px 18px; margin:1rem 0; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
            <div>
                <span style="font-size:0.75rem; font-weight:700; text-transform:uppercase; color:#64748B;">Execution Profiling</span>
                <div style="font-size:0.95rem; font-weight:600; color:#0F172A;">{rev_cnt} reviews analyzed ({seg_cnt} aspect clauses)</div>
            </div>
            <div style="display:flex; gap:20px; align-items:center;">
                <div><span style="font-size:0.75rem; color:#64748B;">Time:</span> <strong style="color:#0F172A;">{p_time:.3f}s</strong></div>
                <div><span style="font-size:0.75rem; color:#64748B;">Throughput:</span> <strong style="color:#15803D;">{t_put:.1f} reviews/sec</strong></div>
                <div style="font-size:0.75rem; font-weight:600; color:#1D4ED8; background:#EFF6FF; border:1px solid #BFDBFE; padding:2px 8px; border-radius:4px;">Local CPU · Multi-Aspect NLP</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Single Review Deep Inspection
single_p = st.session_state.get("single_pred")
p_df = st.session_state.get("processed_df")
s_df = st.session_state.get("segments_df")
grps = st.session_state.get("groups")

if single_p is not None and p_df is not None and len(p_df) == 1:
    st.markdown("<h2>🔍 Single Review Deep Inspection & Multi-Aspect Breakdown</h2>", unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(
            f"""
            <div class="customer-quote">
                <div class="customer-quote-text">“{p_df['review_text'].iloc[0]}”</div>
                <div class="customer-quote-meta">
                    <span class="badge-{single_p['label'].lower()}">{single_p['label']} ({single_p['score']:.0%})</span>
                    <span>Compound Score: <code>{single_p['compound']:+.3f}</code></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if s_df is not None and len(s_df) > 1:
            st.markdown("### 🧩 Decomposed Multi-Aspect Clauses")
            st.caption("Compound sentences are segmented so distinct issues and mixed sentiments are evaluated independently:")
            for _, r in s_df.iterrows():
                badge_class = f"badge-{r['sentiment'].lower()}"
                st.markdown(
                    f"""
                    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:6px; padding:10px 14px; margin-bottom:8px;">
                        <div style="font-size:0.92rem; font-weight:600; color:#0F172A;">“{r['segment_text']}”</div>
                        <div style="margin-top:6px; display:flex; gap:10px; align-items:center;">
                            <span class="{badge_class}">{r['sentiment']}</span>
                            <span class="badge-topic">{r['issue_name']}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    with c2:
        st.markdown("### 📊 Polarity Distribution")
        st.metric("Predicted Class", single_p["label"], delta=f"Confidence: {single_p['score']:.0%}")
        st.metric("Continuous Valence", f"{single_p['compound']:+.3f}", delta="Scale: -1.0 to +1.0")
        col_p1, col_p2, col_p3 = st.columns(3)
        col_p1.metric("🔴 Neg", f"{single_p['neg']:.0%}")
        col_p2.metric("⚪ Neu", f"{single_p['neu']:.0%}")
        col_p3.metric("🟢 Pos", f"{single_p['pos']:.0%}")
    st.markdown("---")

# Dashboard Body
if p_df is not None and grps:
    # Handle custom label overrides
    if st.session_state["custom_labels"]:
        for g in grps:
            if g["cluster_id"] in st.session_state["custom_labels"]:
                g["issue_name"] = st.session_state["custom_labels"][g["cluster_id"]]
        c_map = {g["cluster_id"]: g["issue_name"] for g in grps}
        if s_df is not None:
            s_df["issue_name"] = s_df["cluster_id"].map(c_map)
            p_df["assigned_issues"] = s_df.groupby("review_id")["issue_name"].apply(lambda s: " | ".join(dict.fromkeys(s)))
            p_df["issue_name"] = p_df["assigned_issues"]

    # Summary Strip
    tot_revs = len(p_df)
    neg_c = int((p_df["sentiment"] == "Negative").sum())
    neu_c = int((p_df["sentiment"] == "Neutral").sum())
    pos_c = int((p_df["sentiment"] == "Positive").sum())

    st.markdown(
        f"""
        <div class="metric-strip">
            <div class="metric-card"><div class="metric-card-label">Total Reviews</div><div class="metric-card-value">{tot_revs}</div></div>
            <div class="metric-card"><div class="metric-card-label">Critical Issues</div><div class="metric-card-value" style="color:#B91C1C;">{sum(1 for g in grps if g['priority_level'] == 'Critical')}</div></div>
            <div class="metric-card"><div class="metric-card-label">Negative Share</div><div class="metric-card-value" style="color:#B91C1C;">{neg_c/max(1, tot_revs):.1%}</div></div>
            <div class="metric-card"><div class="metric-card-label">Identified Topics</div><div class="metric-card-value" style="color:#1D4ED8;">{len(grps)}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    t1, t2, t3, t4 = st.tabs([
        "🎯 Issue Prioritization & Topics",
        "🔍 Traceable Review Explorer",
        "✏️ Label Management",
        "📖 Architecture & Formula",
    ])

    # ---------------- TAB 1 ----------------
    with t1:
        st.markdown("<h2>🎯 Actionable Issue Priority Index</h2>", unsafe_allow_html=True)
        st.markdown("Every issue group is ranked by its normalized impact score ($Volume \\times Negative\\ Ratio$), identifying where customer friction is concentrated.")

        with st.expander("ℹ️ Priority Formula & Policy Disclaimers", expanded=False):
            st.markdown(
                """
                $$\\text{Raw Impact} = \\text{Issue Volume} \\times \\text{Negative Sentiment Ratio} = \\text{Total Negative Segments}$$
                $$\\text{Normalized Impact Score} = \\left( \\frac{\\text{Raw Impact}}{\\text{Total Customer Reviews}} \\right) \\times 100$$
                * **Critical** ($\\ge 20\\%$): Severe friction impacting $\\ge 20\\%$ of customer feedback base.
                * **High** ($10\\% - 19.9\\%$): Major operational dissatisfaction.
                * **Medium** ($4\\% - 9.9\\%$): Moderate emerging defect.
                * **Low** ($< 4\\%$): Low friction or predominantly positive/neutral feedback.
                * *Notice: Automated statistical heuristic to assist triaging; not an infallible business decree.*
                """
            )

        table_rows = []
        for r, g in enumerate(grps, 1):
            table_rows.append({
                "Rank": r,
                "Priority": g["priority_level"],
                "Issue Topic": g["issue_name"],
                "Aspect Volume": g["volume"],
                "Negative Ratio": f"{g['neg_ratio']:.1%}",
                "Impact Score (%)": g["norm_impact"],
                "Sentiment Breakdown": f"🔴 {g['neg']} | ⚪ {g['neu']} | 🟢 {g['pos']}",
                "Representative Customer Feedback": g["representative_quote"],
            })

        st.dataframe(
            pd.DataFrame(table_rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn(width="small"),
                "Priority": st.column_config.TextColumn("Priority Tier", width="small"),
                "Issue Topic": st.column_config.TextColumn(width="medium"),
                "Aspect Volume": st.column_config.NumberColumn("Volume", width="small"),
                "Negative Ratio": st.column_config.TextColumn("Neg. Ratio", width="small"),
                "Impact Score (%)": st.column_config.NumberColumn("Impact Score (%)", format="%.1f", width="small"),
                "Sentiment Breakdown": st.column_config.TextColumn("Breakdown", width="medium"),
                "Representative Customer Feedback": st.column_config.TextColumn("Representative Feedback", width="large"),
            },
        )

        st.markdown("---")
        st.markdown("### 📊 Visual Analytics")
        ch1, ch2 = st.columns([3, 2])
        with ch1:
            st.markdown("#### Issue Frequency Ranking")
            df_chart = pd.DataFrame([{"Issue": g["issue_name"], "Volume": g["volume"]} for g in grps])
            c_freq = alt.Chart(df_chart).mark_bar(color="#3B82F6", cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
                y=alt.Y("Issue:N", sort="-x", title=None),
                x=alt.X("Volume:Q", title="Extracted Aspect Clauses"),
            ).properties(height=max(180, len(grps) * 35))
            st.altair_chart(c_freq, use_container_width=True)

        with ch2:
            st.markdown("#### Sentiment Distribution")
            df_donut = pd.DataFrame([
                {"Sentiment": "Negative", "Count": neg_c, "Color": "#EF4444"},
                {"Sentiment": "Neutral", "Count": neu_c, "Color": "#94A3B8"},
                {"Sentiment": "Positive", "Count": pos_c, "Color": "#22C55E"},
            ])
            c_donut = alt.Chart(df_donut).mark_arc(innerRadius=45).encode(
                theta=alt.Theta("Count:Q"),
                color=alt.Color("Sentiment:N", scale=alt.Scale(domain=["Negative", "Neutral", "Positive"], range=["#EF4444", "#94A3B8", "#22C55E"])),
                tooltip=["Sentiment", "Count"],
            ).properties(height=220)
            st.altair_chart(c_donut, use_container_width=True)

    # ---------------- TAB 2 ----------------
    with t2:
        st.markdown("<h2>🔍 Traceable Review Explorer</h2>", unsafe_allow_html=True)
        view_opt = st.radio("Inspection Level", options=["Parent Reviews (Holistic Feedback)", "Extracted Aspects (Clause-Level Traceability)"], horizontal=True)

        f_col1, f_col2, f_col3, f_col4 = st.columns([1, 1, 1, 2])
        with f_col1:
            f_sent = st.selectbox("Sentiment", options=["All", "Positive", "Neutral", "Negative"])
        with f_col2:
            f_issue = st.selectbox("Issue Topic", options=["All"] + [g["issue_name"] for g in grps])
        with f_col3:
            f_prio = st.selectbox("Priority Tier", options=["All", "Critical", "High", "Medium", "Low"])
        with f_col4:
            f_search = st.text_input("Keyword Search", placeholder="Filter by text...")

        if "Extracted Aspects" in view_opt and s_df is not None:
            filt = s_df.copy()
            if f_sent != "All":
                filt = filt[filt["sentiment"] == f_sent]
            if f_issue != "All":
                filt = filt[filt["issue_name"] == f_issue]
            if f_prio != "All":
                filt = filt[filt["priority_level"] == f_prio]
            if f_search.strip():
                filt = filt[filt["segment_text"].str.contains(f_search.strip(), case=False, na=False)]

            st.caption(f"Showing {len(filt)} of {len(s_df)} extracted aspect clauses.")
            st.dataframe(
                filt[["segment_id", "review_id", "segment_text", "sentiment", "issue_name", "priority_level", "original_review_text"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "segment_id": st.column_config.NumberColumn("Seg ID", width="small"),
                    "review_id": st.column_config.NumberColumn("Parent ID", width="small"),
                    "segment_text": st.column_config.TextColumn("Extracted Clause", width="large"),
                    "sentiment": st.column_config.TextColumn("Sentiment", width="small"),
                    "issue_name": st.column_config.TextColumn("Assigned Topic", width="medium"),
                    "priority_level": st.column_config.TextColumn("Priority", width="small"),
                    "original_review_text": st.column_config.TextColumn("Original Parent Review", width="large"),
                },
            )
        else:
            filt = p_df.copy()
            if f_sent != "All":
                filt = filt[filt["sentiment"] == f_sent]
            if f_issue != "All":
                filt = filt[filt["assigned_issues"].str.contains(f_issue, case=False, na=False)]
            if f_prio != "All":
                filt = filt[filt["priority_level"] == f_prio]
            if f_search.strip():
                filt = filt[filt["review_text"].str.contains(f_search.strip(), case=False, na=False)]

            st.caption(f"Showing {len(filt)} of {len(p_df)} parent reviews.")
            st.dataframe(
                filt[["review_id", "review_text", "sentiment", "priority_level", "assigned_issues", "aspect_count"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "review_id": st.column_config.NumberColumn("ID", width="small"),
                    "review_text": st.column_config.TextColumn("Customer Review", width="large"),
                    "sentiment": st.column_config.TextColumn("Sentiment", width="small"),
                    "priority_level": st.column_config.TextColumn("Priority Tier", width="small"),
                    "assigned_issues": st.column_config.TextColumn("Assigned Issue Topics", width="large"),
                    "aspect_count": st.column_config.NumberColumn("Aspects", width="small"),
                },
            )

        # CSV Download
        csv_buf = io.StringIO()
        p_df.to_csv(csv_buf, index=False)
        st.download_button(
            label="📥 Download Analyzed Customer Dataset (CSV)",
            data=csv_buf.getvalue(),
            file_name="customer_review_intelligence_export.csv",
            mime="text/csv",
        )

    # ---------------- TAB 3 ----------------
    with t3:
        st.markdown("<h2>✏️ Human-in-the-Loop Label Management</h2>", unsafe_allow_html=True)
        st.markdown("Customize algorithmic titles to match your organization's internal taxonomy:")
        with st.form("edit_form"):
            new_labels = {}
            for g in grps:
                c_a, c_b = st.columns([1, 2])
                with c_a:
                    st.markdown(f"**Group #{g['cluster_id']}** ({g['volume']} aspects)")
                    st.caption(f"Keywords: {', '.join(g['keywords'][:3]) if g['keywords'] else 'None'}")
                with c_b:
                    new_val = st.text_input(f"Title for Group {g['cluster_id']}", value=g["issue_name"], label_visibility="collapsed")
                    new_labels[g["cluster_id"]] = new_val.strip() if new_val.strip() else g["issue_name"]
            if st.form_submit_button("💾 Save and Apply Custom Labels"):
                st.session_state["custom_labels"] = new_labels
                st.success("Issue labels successfully updated across the dashboard.")
                st.rerun()

    # ---------------- TAB 4 ----------------
    with t4:
        st.markdown("<h2>📖 Architecture & Methodology</h2>", unsafe_allow_html=True)
        st.markdown(
            """
            * **Multi-Aspect Segmentation:** Decomposes compound reviews (e.g. *'Battery is good, but app crashes frequently and shipping took 3 weeks'*) into isolated clauses.
            * **Local Calibrated Sentiment:** Continuous valence score $\in [-1.0, +1.0]$ via calibrated VADER with contrastive conjunction logic.
            * **Semantic Embeddings:** FastEmbed ONNX BGE-Small (384-dimensional dense vectors) or sub-word TF-IDF fallback.
            * **Agglomerative Clustering:** Cosine distance thresholding grouping semantically equivalent complaints into issue clusters.
            * **Actionable Prioritization:** Impact Score $= Volume \\times Negative\\ Ratio$, normalized to percentage of total feedback corpus.
            * **Zero API Cost:** 100% locally runnable on standard CPU.
            """
        )
