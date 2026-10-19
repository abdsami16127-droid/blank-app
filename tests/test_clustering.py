"""Unit tests for semantic embeddings, clustering, and issue extraction."""

import numpy as np
import pytest
import pandas as pd

from src.clustering import SemanticClusterer
from src.embeddings import EmbeddingEngine
from src.issue_extractor import IssueExtractor, format_issue_title


@pytest.fixture(scope="module")
def embedding_engine():
    return EmbeddingEngine()


def test_embedding_generation_and_normalization(embedding_engine):
    """Verify that embeddings are correctly generated with unit-normalized vectors."""
    texts = [
        "Battery drains within 2 hours.",
        "The charge does not last at all.",
        "Package arrived two weeks late.",
    ]
    embeddings = embedding_engine.embed_texts(texts)

    assert embeddings.shape == (3, 384)
    # Check L2 normalization (norms should be 1.0)
    norms = np.linalg.norm(embeddings, axis=1)
    for norm in norms:
        assert pytest.approx(norm, rel=1e-3) == 1.0


def test_semantic_similarity(embedding_engine):
    """Verify semantic equivalence captures different wordings of the same problem."""
    # Two battery complaints with completely different words
    bat1 = embedding_engine.embed_single("The battery drains extremely fast.")
    bat2 = embedding_engine.embed_single("My phone dies within hours of being unplugged.")
    # An unrelated shipping complaint
    ship = embedding_engine.embed_single("The delivery was delayed by three weeks.")

    sim_battery_pair = np.dot(bat1, bat2)
    sim_battery_shipping = np.dot(bat1, ship)

    # Battery pair must be much closer in embedding space than battery to shipping
    assert sim_battery_pair > sim_battery_shipping
    assert sim_battery_pair > 0.60


def test_semantic_clustering_separates_distinct_issues(embedding_engine):
    """Verify that distinct issues are partitioned into separate clusters."""
    texts = [
        # Battery issues
        "Battery drains in less than 4 hours.",
        "Charge does not last through my commute.",
        "Phone dies quickly after charging.",
        "Battery percentage drops 20% in 15 minutes.",
        # Shipping issues
        "Package arrived two weeks late.",
        "Shipping tracking was delayed for days.",
        "Late delivery ruined my anniversary gift.",
        "Delivery was delayed with no updates.",
    ]
    embeddings = embedding_engine.embed_texts(texts)
    clusterer = SemanticClusterer(method="agglomerative", distance_threshold=0.45)
    labels = clusterer.fit_predict(embeddings)

    assert len(labels) == 8
    # Battery reviews (0..3) should share the same cluster
    assert labels[0] == labels[1] == labels[2] == labels[3]
    # Shipping reviews (4..7) should share the same cluster
    assert labels[4] == labels[5] == labels[6] == labels[7]
    # Battery cluster must differ from shipping cluster
    assert labels[0] != labels[4]


def test_cluster_medoid_selection():
    """Verify that medoid identifies the central review in a cluster."""
    # Synthetic 2D vectors: medoid should be vector closest to mean
    embeddings = np.array([
        [1.0, 0.0],
        [0.99, 0.01],
        [0.70, 0.70],
    ])
    labels = np.array([0, 0, 1])
    medoids = SemanticClusterer.find_cluster_medoids(embeddings, labels)

    assert 0 in medoids
    assert 1 in medoids
    assert medoids[0] in [0, 1]
    assert medoids[1] == 2


def test_issue_extraction_and_dataframe_enrichment(embedding_engine):
    """Verify end-to-end issue extraction, keyword assignment, and DataFrame enrichment."""
    data = {
        "review_id": list(range(6)),
        "review_text": [
            "Battery runs out very quickly.",
            "Battery percentage drops fast.",
            "Phone dies after 2 hours.",
            "Shipping was very late.",
            "Package arrived after huge delay.",
            "Delivery courier was delayed.",
        ],
        "sentiment": ["Negative", "Negative", "Negative", "Negative", "Negative", "Negative"],
    }
    df = pd.DataFrame(data)
    embeddings = embedding_engine.embed_texts(df["review_text"].tolist())

    clusterer = SemanticClusterer(method="agglomerative", distance_threshold=0.45)
    labels = clusterer.fit_predict(embeddings)

    extractor = IssueExtractor()
    issue_groups = extractor.extract_issues(df, embeddings, labels)

    assert len(issue_groups) == 2
    # Verify groups contain relevant keywords
    group_titles = [g.issue_name.lower() for g in issue_groups]
    assert any("battery" in t or "dies" in t for t in group_titles)
    assert any("shipping" in t or "delay" in t or "package" in t for t in group_titles)

    # Enrich DataFrame
    enriched_df = extractor.assign_issue_labels_to_dataframe(df, issue_groups, labels)
    assert "cluster_id" in enriched_df.columns
    assert "issue_name" in enriched_df.columns
    assert len(enriched_df["issue_name"].unique()) == 2


def test_format_issue_title():
    """Test title formatting edge cases."""
    assert format_issue_title(["battery life", "drain"]) == "Battery Life / Drain"
    assert format_issue_title(["charging"]) == "Charging"
    assert format_issue_title([], fallback_text="Screen cracked on first drop") == "Screen Cracked On First Drop"
