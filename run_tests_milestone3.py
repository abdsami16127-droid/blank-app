"""Milestone 3 Comprehensive Verification Test Suite."""

import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd

from src.clustering import SemanticClusterer
from src.embeddings import EmbeddingEngine
from src.issue_extractor import IssueExtractor, format_issue_title


def test_embedding_engine():
    print("1. Testing EmbeddingEngine...", flush=True)
    engine = EmbeddingEngine()
    texts = [
        "Battery drains in less than 4 hours.",
        "Charge does not last through my commute.",
        "Phone dies quickly after charging.",
        "Package arrived two weeks late.",
        "Shipping tracking was delayed for days.",
    ]
    embs = engine.embed_texts(texts)
    assert embs.shape == (5, 384), f"Expected shape (5, 384), got {embs.shape}"
    norms = np.linalg.norm(embs, axis=1)
    for n in norms:
        assert abs(n - 1.0) < 1e-3, f"Vector not normalized: {n}"
    print("   ✓ Dense embedding generation & unit normalization verified", flush=True)

    # Semantic similarity check
    bat1 = engine.embed_single("Battery drains too quickly.")
    bat2 = engine.embed_single("The charge doesn't last.")
    ship = engine.embed_single("Delivery was delayed by two weeks.")
    sim_bat = float(np.dot(bat1, bat2))
    sim_ship = float(np.dot(bat1, ship))
    assert sim_bat > sim_ship, f"Battery similarity ({sim_bat}) should exceed cross-topic ({sim_ship})"
    assert sim_bat > 0.60, f"Expected similarity > 0.60, got {sim_bat}"
    print(f"   ✓ Semantic equivalence verified (Battery pair: {sim_bat:.3f} vs Cross-topic: {sim_ship:.3f})", flush=True)
    return engine, embs, texts


def test_clustering(engine, embs, texts):
    print("\n2. Testing SemanticClusterer...", flush=True)
    clusterer = SemanticClusterer(method="agglomerative", distance_threshold=0.45)
    labels = clusterer.fit_predict(embs)
    assert len(labels) == 5
    # Battery group (0, 1, 2) must share the same cluster
    assert labels[0] == labels[1] == labels[2], f"Battery reviews split: {labels[:3]}"
    # Shipping group (3, 4) must share the same cluster
    assert labels[3] == labels[4], f"Shipping reviews split: {labels[3:]}"
    # Must be distinct clusters
    assert labels[0] != labels[3], f"Battery and shipping merged into cluster {labels[0]}"
    print(f"   ✓ Semantic grouping verified (Cluster assignments: {labels})", flush=True)

    # Medoid test
    medoids = SemanticClusterer.find_cluster_medoids(embs, labels)
    assert len(medoids) == 2
    print(f"   ✓ Central cluster medoids identified: {medoids}", flush=True)

    # Metrics
    metrics = SemanticClusterer.evaluate_clustering(embs, labels)
    assert metrics["cluster_count"] == 2
    assert metrics["silhouette_score"] > 0.0
    print(f"   ✓ Clustering silhouette score: {metrics['silhouette_score']}", flush=True)
    return labels


def test_issue_extraction(engine, embs, texts, labels):
    print("\n3. Testing IssueExtractor...", flush=True)
    df = pd.DataFrame({
        "review_id": list(range(len(texts))),
        "review_text": texts,
        "sentiment": ["Negative"] * len(texts),
    })
    extractor = IssueExtractor()
    groups = extractor.extract_issues(df, embs, labels)
    assert len(groups) == 2

    # Verify titles contain descriptive keywords
    titles = [g.issue_name.lower() for g in groups]
    assert any("battery" in t or "drain" in t or "charge" in t or "phone" in t for t in titles)
    assert any("shipping" in t or "package" in t or "week" in t or "delay" in t for t in titles)

    for g in groups:
        print(f"   * Group {g.cluster_id}: '{g.issue_name}' | Count: {g.review_count} ({g.percentage}%) | Medoid Quote: \"{g.representative_review}\"", flush=True)

    # Verify DataFrame assignment
    enriched = extractor.assign_issue_labels_to_dataframe(df, groups, labels)
    assert "cluster_id" in enriched.columns
    assert "issue_name" in enriched.columns
    assert len(enriched["issue_name"].unique()) == 2
    print("   ✓ Traceable DataFrame enrichment verified", flush=True)


if __name__ == "__main__":
    print("==================================================", flush=True)
    print("     CUSTOMER REVIEW INTELLIGENCE SYSTEM          ", flush=True)
    print("         MILESTONE 3 VERIFICATION SUITE           ", flush=True)
    print("==================================================", flush=True)
    engine, embs, texts = test_embedding_engine()
    labels = test_clustering(engine, embs, texts)
    test_issue_extraction(engine, embs, texts, labels)
    print("\n==================================================", flush=True)
    print("   ALL MILESTONE 3 TESTS PASSED (100% SUCCESS)    ", flush=True)
    print("==================================================", flush=True)
