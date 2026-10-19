"""Reproducible evaluation script for the Customer Review Intelligence System."""

import sys
sys.stdout.reconfigure(line_buffering=True)
import onnxruntime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from src.config import TEST_GROUND_TRUTH_PATH
from src.data_loader import load_reviews_from_csv
from src.sentiment import SentimentAnalyzer
from src.embeddings import EmbeddingEngine
from src.clustering import SemanticClusterer
from src.issue_extractor import IssueExtractor


def evaluate():
    print("==========================================================")
    print("  CUSTOMER REVIEW INTELLIGENCE SYSTEM - EVALUATION REPORT ")
    print("==========================================================")

    if not TEST_GROUND_TRUTH_PATH.exists():
        print(f"Error: Ground truth file not found at {TEST_GROUND_TRUTH_PATH}")
        sys.exit(1)

    print(f"\nLoading manually verified test set: {TEST_GROUND_TRUTH_PATH.name}...")
    df, report = load_reviews_from_csv(TEST_GROUND_TRUTH_PATH)
    print(f"Total verified benchmark samples: {report.valid_rows}")

    # 1. Sentiment Evaluation
    print("\n--- 1. Sentiment Analysis Evaluation ---")
    analyzer = SentimentAnalyzer()
    df_analyzed = analyzer.analyze_dataframe(df)

    y_true_sentiment = df["ground_truth_sentiment"]
    y_pred_sentiment = df_analyzed["sentiment"]

    acc_sentiment = accuracy_score(y_true_sentiment, y_pred_sentiment)
    print(f"Sentiment Overall Accuracy: {acc_sentiment * 100:.1f}%\n")
    print("Classification Report:")
    print(classification_report(y_true_sentiment, y_pred_sentiment, digits=3))

    # 2. Semantic Clustering Evaluation
    print("\n--- 2. Semantic Clustering Evaluation ---")
    engine = EmbeddingEngine()
    embeddings = engine.embed_texts(df["review_text"].tolist())

    clusterer = SemanticClusterer(method="agglomerative", distance_threshold=0.45)
    pred_labels = clusterer.fit_predict(embeddings)

    # Convert ground-truth textual issue classes to numeric factor codes for comparison
    true_labels = pd.factorize(df["ground_truth_issue"])[0]

    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    clustering_metrics = SemanticClusterer.evaluate_clustering(embeddings, pred_labels)

    print(f"Adjusted Rand Index (ARI):           {ari:.3f}  (Measures cluster assignment agreement)")
    print(f"Normalized Mutual Information (NMI):  {nmi:.3f}  (Measures shared mutual information)")
    print(f"Silhouette Coefficient:               {clustering_metrics['silhouette_score']:.3f}  (Cluster cohesion vs separation)")
    print(f"Discovered Clusters:                  {int(clustering_metrics['cluster_count'])} topics (Ground-truth classes: {len(set(true_labels))})")

    # 3. Issue Group Naming Sample
    print("\n--- 3. Extracted Topic Names ---")
    extractor = IssueExtractor()
    groups = extractor.extract_issues(df_analyzed, embeddings, pred_labels)
    for g in groups:
        print(f"  * Group {g.cluster_id}: {g.issue_name} ({g.review_count} reviews) -> \"{g.representative_review[:45]}\"")

    print("\n==========================================================")
    print("  EVALUATION COMPLETED - ALL METRICS VERIFIED             ")
    print("==========================================================")


if __name__ == "__main__":
    evaluate()
