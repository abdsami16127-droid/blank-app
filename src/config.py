"""Configuration settings, constants, and defaults for the Customer Review Intelligence System."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_CSV_PATH = DATA_DIR / "sample_reviews.csv"
TEST_GROUND_TRUTH_PATH = DATA_DIR / "test_ground_truth.csv"

# Ingestion & Validation
CANDIDATE_REVIEW_COLUMNS = [
    "review",
    "review_text",
    "review_body",
    "reviews",
    "feedback",
    "comment",
    "comments",
    "text",
    "customer_review",
    "content",
    "body",
    "complaint",
    "issue_description",
]

MIN_REVIEW_CHAR_LENGTH = 5
MAX_REVIEW_CHAR_LENGTH = 5000

# Sentiment Analysis
SENTIMENT_POSITIVE = "Positive"
SENTIMENT_NEUTRAL = "Neutral"
SENTIMENT_NEGATIVE = "Negative"
SENTIMENT_LABELS = [SENTIMENT_POSITIVE, SENTIMENT_NEUTRAL, SENTIMENT_NEGATIVE]

# Clustering & NLP
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MIN_CLUSTER_SIZE_DEFAULT = 3
NOISE_CLUSTER_LABEL = -1
NOISE_CLUSTER_NAME = "Uncategorized / General Feedback"

# Multi-Aspect Segmentation Constants
MIN_SEGMENT_CHAR_LENGTH = 10
MIN_SEGMENT_WORD_COUNT = 3

# Issue Priority Index Thresholds (Normalized Impact Score = (Negative Count / Total Corpus Size) * 100)
PRIORITY_CRITICAL = "Critical"
PRIORITY_HIGH = "High"
PRIORITY_MEDIUM = "Medium"
PRIORITY_LOW = "Low"

PRIORITY_THRESHOLDS = {
    PRIORITY_CRITICAL: 20.0,  # >= 20.0% of total customer feedback base complaining about this issue
    PRIORITY_HIGH: 10.0,      # >= 10.0% and < 20.0%
    PRIORITY_MEDIUM: 4.0,     # >= 4.0% and < 10.0%
    PRIORITY_LOW: 0.0,        # < 4.0%
}

# Editorial UI Palette (Restrained, Professional, Non-flashy)
COLOR_PALETTE = {
    "background_subtle": "#F8FAFC",
    "border_subtle": "#E2E8F0",
    "border_strong": "#CBD5E1",
    "text_primary": "#0F172A",
    "text_secondary": "#475569",
    "text_muted": "#64748B",
    "positive": "#15803D",        # Forest green
    "positive_subtle": "#F0FDF4",
    "neutral": "#475569",         # Slate
    "neutral_subtle": "#F1F5F9",
    "negative": "#B91C1C",        # Crimson
    "negative_subtle": "#FEF2F2",
    "accent": "#1E293B",          # Deep slate
}
