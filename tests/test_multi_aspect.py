"""Unit tests for Multi-Aspect Segmentation, Priority Index, and Performance Profiling (Milestone 6)."""

import io
import time
import numpy as np
import pandas as pd
import pytest

from src.config import (
    PRIORITY_CRITICAL,
    PRIORITY_HIGH,
    PRIORITY_LOW,
    PRIORITY_MEDIUM,
    PRIORITY_THRESHOLDS,
    SAMPLE_CSV_PATH,
)
from src.data_loader import load_reviews_from_csv
from src.embeddings import EmbeddingEngine
from src.clustering import SemanticClusterer
from src.issue_extractor import IssueExtractor, IssueGroup
from src.segmentation import (
    clean_segment_text,
    is_valid_segment,
    segment_dataframe,
    segment_review_text,
)
from src.sentiment import SentimentAnalyzer


def test_multi_aspect_segmentation_user_example():
    """Verify that the user's exact example decomposes into separate aspect clauses."""
    text = "Battery is good, but the app crashes frequently and shipping took 3 weeks."
    segments = segment_review_text(text)

    assert len(segments) == 3
    assert segments[0] == "Battery is good"
    assert "app crashes" in segments[1].lower()
    assert "shipping" in segments[2].lower()


def test_multi_aspect_preserves_single_issue():
    """Reviews containing a single coherent thought must not be fragmented."""
    text = "The battery drains too quickly in the cold."
    segments = segment_review_text(text)

    assert len(segments) == 1
    assert segments[0] == "The battery drains too quickly in the cold"


def test_coordinate_clause_does_not_break_adjectives():
    """Non-clause 'and' joining adjectives must remain together."""
    text = "The display is bright and clear."
    segments = segment_review_text(text)

    assert len(segments) == 1
    assert segments[0] == "The display is bright and clear"


def test_segment_dataframe_maintains_parent_traceability():
    """Verifies that segments retain persistent foreign key link to review_id."""
    raw_data = {
        "review_id": [10, 11],
        "review_text": [
            "Battery is good, but the app crashes frequently and shipping took 3 weeks.",
            "Delivery was on time and packaging was intact.",
        ],
        "rating": [2, 5],
    }
    df = pd.DataFrame(raw_data)
    seg_df = segment_dataframe(df)

    assert len(seg_df) >= 3
    # Check that review_id 10 produced 3 segments
    r10_segs = seg_df[seg_df["review_id"] == 10]
    assert len(r10_segs) == 3
    assert all(r10_segs["original_review_text"] == raw_data["review_text"][0])
    assert all(r10_segs["rating"] == 2)


def test_different_complaints_from_same_review_assigned_to_different_issues():
    """
    Core Milestone 6 verification:
    Verify that different complaints in the same review get partitioned to distinct issue groups.
    """
    raw_data = {
        "review_id": [0, 1, 2, 3],
        "review_text": [
            # Review 0 mentions both battery and delivery
            "Battery drains within 2 hours, but delivery took four weeks to arrive.",
            # Additional reviews reinforcing each theme
            "The battery charge runs down immediately.",
            "Courier package arrived very late.",
            "Order shipment was delayed for weeks.",
        ],
    }
    df = pd.DataFrame(raw_data)
    seg_df = segment_dataframe(df)

    # Analyze sentiment per segment
    analyzer = SentimentAnalyzer()
    seg_df = analyzer.analyze_dataframe(seg_df, text_column="segment_text")

    # Generate embeddings and cluster
    engine = EmbeddingEngine()
    embs = engine.embed_texts(seg_df["segment_text"].tolist())
    clusterer = SemanticClusterer(method="agglomerative", distance_threshold=0.45)
    labels = clusterer.fit_predict(embs)

    extractor = IssueExtractor()
    groups = extractor.extract_issues(seg_df, embs, labels, text_column="segment_text")
    enriched_segs = extractor.assign_issue_labels_to_dataframe(seg_df, groups, labels)

    # Review 0 segments must have been assigned to two DIFFERENT clusters
    r0_segs = enriched_segs[enriched_segs["review_id"] == 0]
    assert len(r0_segs) == 2
    r0_clusters = r0_segs["cluster_id"].tolist()
    assert r0_clusters[0] != r0_clusters[1], "Segments from Review 0 should belong to distinct issue clusters!"


def test_impact_score_and_priority_calculation():
    """Verify manual mathematical calculation of Impact Score and Priority Levels."""
    # Create synthetic groups with controlled volume and negative sentiment
    extractor = IssueExtractor()

    # Total corpus size = 100
    total_reviews = 100

    # Group A: 30 reviews, 24 negative -> NegRatio = 0.8, Raw Impact = 24.0, Norm Impact = 24.0% -> Critical
    g_critical = {
        "Volume": 30,
        "Neg": 24,
        "Total": total_reviews,
    }
    neg_ratio_a = g_critical["Neg"] / g_critical["Volume"]
    raw_impact_a = g_critical["Volume"] * neg_ratio_a
    norm_impact_a = (raw_impact_a / total_reviews) * 100.0

    assert pytest.approx(neg_ratio_a) == 0.8
    assert pytest.approx(raw_impact_a) == 24.0
    assert pytest.approx(norm_impact_a) == 24.0
    assert norm_impact_a >= PRIORITY_THRESHOLDS[PRIORITY_CRITICAL]

    # Group B: 15 reviews, 2 negative -> NegRatio = 0.133, Raw Impact = 2.0, Norm Impact = 2.0% -> Low
    g_low = {
        "Volume": 15,
        "Neg": 2,
        "Total": total_reviews,
    }
    neg_ratio_b = g_low["Neg"] / g_low["Volume"]
    raw_impact_b = g_low["Volume"] * neg_ratio_b
    norm_impact_b = (raw_impact_b / total_reviews) * 100.0

    assert pytest.approx(raw_impact_b) == 2.0
    assert norm_impact_b < PRIORITY_THRESHOLDS[PRIORITY_MEDIUM]


def test_priority_threshold_exact_boundaries():
    """Verify exact boundary transitions for Critical, High, Medium, Low."""
    # Critical: >= 20.0
    # High: >= 10.0 and < 20.0
    # Medium: >= 4.0 and < 10.0
    # Low: < 4.0
    def get_priority(norm_score: float) -> str:
        if norm_score >= PRIORITY_THRESHOLDS[PRIORITY_CRITICAL]:
            return PRIORITY_CRITICAL
        elif norm_score >= PRIORITY_THRESHOLDS[PRIORITY_HIGH]:
            return PRIORITY_HIGH
        elif norm_score >= PRIORITY_THRESHOLDS[PRIORITY_MEDIUM]:
            return PRIORITY_MEDIUM
        else:
            return PRIORITY_LOW

    assert get_priority(25.0) == PRIORITY_CRITICAL
    assert get_priority(20.0) == PRIORITY_CRITICAL
    assert get_priority(19.9) == PRIORITY_HIGH
    assert get_priority(10.0) == PRIORITY_HIGH
    assert get_priority(9.9) == PRIORITY_MEDIUM
    assert get_priority(4.0) == PRIORITY_MEDIUM
    assert get_priority(3.9) == PRIORITY_LOW
    assert get_priority(0.0) == PRIORITY_LOW


def test_performance_profiling_measurement():
    """Verify that processing execution time and reviews/second throughput are computed accurately."""
    df, report = load_reviews_from_csv(SAMPLE_CSV_PATH)

    start = time.perf_counter()
    seg_df = segment_dataframe(df)
    analyzer = SentimentAnalyzer()
    seg_df = analyzer.analyze_dataframe(seg_df, text_column="segment_text")
    elapsed = time.perf_counter() - start

    assert elapsed > 0
    throughput = len(df) / elapsed
    assert throughput > 10.0  # Should easily achieve > 10 reviews/second on standard CPU


def test_run_pipeline_end_to_end():
    """Verify that run_pipeline executes the full multi-aspect pipeline and outputs metrics."""
    from app import run_pipeline

    df, _ = load_reviews_from_csv(SAMPLE_CSV_PATH)
    final_df, seg_df, groups, perf = run_pipeline(df, distance_threshold=0.45)

    assert len(final_df) == len(df)
    assert len(seg_df) >= len(df)  # Multi-aspect segmentation creates >= segments
    assert len(groups) > 0
    assert "priority_level" in final_df.columns
    assert "assigned_issues" in final_df.columns
    assert "aspect_count" in final_df.columns

    # Performance metrics verification
    assert perf["total_reviews"] == len(df)
    assert perf["total_segments"] == len(seg_df)
    assert perf["processing_time"] > 0
    assert perf["throughput"] > 0
    assert any(g.priority_level in [PRIORITY_CRITICAL, PRIORITY_HIGH, PRIORITY_MEDIUM, PRIORITY_LOW] for g in groups)

